from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .contracts import WorkflowOptions
from .runner import ALL_STAGES, EvaluationWorkflowRunner
from .run_control import WorkflowRunControlError


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the formal ONNX Splitpoint Evaluation Workflow.")
    ap.add_argument("--profile", required=True, help="Evaluation profile id (e.g. smoke, final) or YAML path")
    ap.add_argument("--out", required=True, help="Output root for EvaluationRuns")
    ap.add_argument("--profile-driven", action="store_true", help="Resolve runtime options from the same frozen profile snapshot used by the GUI. Only profile/out/models-root/only-model/execution-mode, start guards, and the attested cache-canary run id may be combined with this mode.")
    ap.add_argument("--require-run-mode", choices=["smoke", "standard", "final"], default="", help="Fail before run creation unless the resolved profile has this run mode.")
    ap.add_argument("--require-fresh-run", action="store_true", help="Fail before run creation if resume, partial, result-import, no-remote, or non-runtime shortcuts are active.")
    ap.add_argument("--models-root", default="", help="Optional root containing exported ONNX files")
    ap.add_argument("--include-reserve", action="store_true", help="Include reserve models from the profile")
    ap.add_argument("--resume", action="store_true", help="Resume the latest matching run or --run-id")
    ap.add_argument("--rerun-generated", action="store_true", help="Reuse generated BenchmarkSets/backend artifacts and rerun only benchmarks, validation, hardware summary and reports with current runner/report code")
    ap.add_argument("--finalize-partial", action="store_true", help="Do not run heavy stages; resume an existing run and regenerate validation/aggregation/report artifacts from completed model folders. Useful after cancelling a late YOLO/Energy job.")
    ap.add_argument("--run-id", default="", help="Explicit run id / run directory name")
    ap.add_argument("--only-model", default="", help="Run a single model id from the profile")
    ap.add_argument("--max-models", type=int, default=None, help="Limit number of models for smoke tests")
    ap.add_argument("--dry-run", action="store_true", help="Write workflow contract artifacts without invoking heavy services")
    ap.add_argument("--skip-analysis", action="store_true", help="Compatibility flag; writes plan artifacts when analysis cannot parse ONNX")
    ap.add_argument("--skip-benchmarks", action="store_true", help="Skip runtime benchmark execution/remote services. With --execution-mode generate_benchmarksets, benchmarksets are still generated.")
    ap.add_argument("--execution-mode", choices=["contracts_only", "generate_benchmarksets", "generate_and_run", "legacy_benchmarkset"], default="", help="Workflow execution depth. v49n uses the existing BenchmarkSet generator as the source of truth for generate_benchmarksets/generate_and_run; legacy_benchmarkset is an explicit alias.")
    ap.add_argument("--benchmark-sets-root", default="", help="Optional external root for generated benchmark sets; defaults inside EvaluationRuns")
    ap.add_argument("--benchmark-results-root", default="", help="Optional directory containing existing benchmark_results_*.json/csv files to ingest")
    ap.add_argument("--result-source", action="append", default=[], help="Extra benchmark result file/folder to ingest (benchmark_results_*.json/csv)")
    ap.add_argument("--benchmark-provider", default="", help="Provider override passed to generated benchmark_suite.py in generate_and_run mode (cpu/cuda/tensorrt/auto)")
    ap.add_argument("--benchmark-warmup", type=int, default=1, help="Warmup iterations for local generated benchmark_suite.py execution")
    ap.add_argument("--benchmark-runs", type=int, default=3, help="Measured iterations for local generated benchmark_suite.py execution")
    ap.add_argument("--benchmark-timeout-s", type=int, default=0, help="Per-case/runner timeout for local generated benchmark execution; 0 disables")
    ap.add_argument("--benchmark-execution-backend", default="auto", choices=["auto", "local", "remote"], help="Execution dispatch preference in generate_and_run mode. auto uses remote for Hailo/remote profiles, local for runnable local suites.")
    ap.add_argument("--benchmark-preset", default="auto", choices=["auto", "classification", "detection"], help="Validation preset passed to generated benchmark_suite.py")
    ap.add_argument("--benchmark-image", default="default", help="Image argument passed to generated benchmark_suite.py")
    ap.add_argument("--benchmark-extra-arg", action="append", default=[], help="Extra argument appended to generated benchmark_suite.py; repeatable")
    ap.add_argument("--hailo-build-mode", choices=["auto", "reuse_only", "reuse_and_build_missing", "reuse_build_missing", "request", "local", "venv", "wsl"], default="reuse_only", help="Formal Hailo build policy. reuse_and_build_missing/auto reuse existing HEFs and queue missing builds explicitly.")
    ap.add_argument("--hailo-hw-arch", default="hailo8", help="Hailo hardware architecture label for queued build requests")
    ap.add_argument("--hailo-targets", default="", help="Comma-separated Hailo targets copied from Hardware/Profile, e.g. hailo8,hailo10n")
    ap.add_argument("--hailo-build-backend", default="auto", help="Optional Hailo build backend/service selector")
    ap.add_argument("--hailo-build-timeout-s", type=int, default=3600, help="Timeout budget for future Hailo build service handoff")
    ap.add_argument("--no-hailo-build-full", action="store_true", help="Do not request/expect a full-model Hailo baseline build")
    ap.add_argument("--no-hailo-build-part1", action="store_true", help="Do not request/expect split part1 Hailo builds")
    ap.add_argument("--no-hailo-build-part2", action="store_true", help="Do not request/expect split part2 Hailo builds")
    ap.add_argument("--hailo-preset", default="quick", help="Hailo build preset copied from Hardware/Profile")
    ap.add_argument("--hailo-opt-level", type=int, default=0, help="Hailo optimization level copied from Hardware/Profile")
    ap.add_argument("--hailo-calib-dir", default="", help="Hailo calibration directory copied from Hardware/Profile")
    ap.add_argument("--hailo-calib-count", type=int, default=16, help="Hailo calibration image count")
    ap.add_argument("--hailo-calib-batch-size", type=int, default=8, help="Hailo calibration batch size")
    ap.add_argument("--hailo-force-build", action="store_true", help="Explicitly request and confirm Hailo Force for this start: bypass compatible HEFs")
    ap.add_argument("--confirm-force-build", action="append", choices=["hailo", "deepx"], default=[], help="Confirm inherited Force for this start only; repeat for each affected backend, including Resume. Does not enable Force.")
    ap.add_argument("--hailo-keep-artifacts", action="store_true", help="Preserve intermediate Hailo build artifacts for diagnostics")
    ap.add_argument("--validation-mode", choices=["summary_only", "strict", "disabled"], default="summary_only", help="v49i validation binding mode. summary_only summarizes suite metrics and writes explicit gaps; strict keeps validation gaps visible; disabled skips validation summaries.")
    ap.add_argument("--validation-max-cases", type=int, default=0, help="Reserved cap for future direct validation adapter execution; 0 summarizes all normalized rows.")
    ap.add_argument("--validation-require-explicit", action="store_true", help="Mark runtime-measured cases without explicit validation metrics as thesis-critical gaps.")
    ap.add_argument("--validation-require-task-metrics", action="store_true", help="For thesis/final detection runs, require task metrics (classification/detection) instead of accepting summary-only rows.")
    ap.add_argument("--validation-max-abs-error-threshold", type=float, default=1e-3, help="Numeric validation threshold for max absolute error when result rows expose it.")
    ap.add_argument("--validation-mean-abs-error-threshold", type=float, default=1e-4, help="Numeric validation threshold for mean absolute error when result rows expose it.")
    ap.add_argument("--validation-cosine-threshold", type=float, default=0.999, help="Numeric validation threshold for cosine similarity when result rows expose it.")
    ap.add_argument("--hardware-smoke-mode", choices=["summary_only", "strict", "disabled"], default="summary_only", help="v49i hardware smoke report mode. summary_only reconciles Hailo/remote/result artifacts; disabled omits active smoke interpretation.")
    ap.add_argument("--hardware-smoke-timeout-s", type=int, default=5, help="Reserved timeout budget for optional lightweight hardware/remote smoke probes.")
    ap.add_argument("--hardware-setups-file", default="", help="Optional hardware_setups.yaml registry. Empty uses ~/.onnx_splitpoint_tool/hardware_setups.yaml.")
    ap.add_argument("--hardware-setup", action="append", default=[], help="Hardware setup id to include, e.g. orin_nx_hailo8_01. Repeatable.")
    ap.add_argument("--hardware-group", action="append", default=[], help="Hardware setup group to include, e.g. all_accelerators. Repeatable.")
    ap.add_argument("--no-remote", action="store_true", help="Do not start remote benchmark services")
    ap.add_argument("--remote-host-id", default="", help="Remote host id to select from --remote-hosts-file or profile remote_execution.hosts")
    ap.add_argument("--remote-hosts-file", default="", help="JSON/YAML file with remote_hosts entries. Secrets are not stored; use SSH config/keys.")
    ap.add_argument("--remote-host-json", default="", help="Serialized remote HostConfig JSON. Usually set by the GUI.")
    ap.add_argument("--remote-host", default="", help="Direct remote hostname/IP for RemoteBenchmarkService")
    ap.add_argument("--remote-user", default="", help="Direct remote SSH user")
    ap.add_argument("--remote-port", type=int, default=22, help="Direct remote SSH port")
    ap.add_argument("--remote-base-dir", default="~/splitpoint_runs", help="Remote base directory for uploaded benchmark suites")
    ap.add_argument("--remote-ssh-extra-args", default="", help="Extra args passed to ssh/scp")
    ap.add_argument("--remote-working-dir", default="", help="Local root for RemoteBenchmarkService Results/ cache; defaults under the model benchmark_results folder")
    ap.add_argument("--remote-venv", default="", help="Optional remote venv/setup snippet, e.g. source ~/hailo_env/bin/activate")
    ap.add_argument("--remote-provider", default="auto", help="Remote provider override passed to benchmark_suite.py")
    ap.add_argument("--remote-warmup", type=int, default=10, help="Remote warmup iterations")
    ap.add_argument("--remote-timeout-s", type=int, default=0, help="Outer remote timeout; 0 lets the remote service choose/raise")
    ap.add_argument("--remote-transfer-mode", default="bundle", choices=["bundle", "direct"], help="Remote suite transfer mode")
    ap.add_argument("--no-remote-reuse-bundle", action="store_true", help="Rebuild/upload the remote bundle instead of reusing the cached suite tarball")
    ap.add_argument("--no-remote-resume", action="store_true", help="Do not resume previous remote runs")
    ap.add_argument("--remote-repeats", type=int, default=1, help="Remote repeat groups")
    ap.add_argument("--remote-iters", type=int, default=0, help="Remote measured iterations. 0 reuses --benchmark-runs.")
    ap.add_argument("--remote-add-arg", action="append", default=[], help="Extra remote benchmark_suite.py argument; repeatable")
    ap.add_argument("--remote-throughput-frames", type=int, default=24)
    ap.add_argument("--remote-throughput-warmup-frames", type=int, default=6)
    ap.add_argument("--remote-throughput-queue-depth", type=int, default=2)
    ap.add_argument("--parallel-remote-setups", dest="parallel_remote_setups", action="store_true", default=None, help="Run independent hardware setup dispatches in parallel (default for multi-setup profiles).")
    ap.add_argument("--no-parallel-remote-setups", dest="parallel_remote_setups", action="store_false", help="Force sequential remote hardware setup dispatch.")
    ap.add_argument("--max-parallel-setups", type=int, default=0, help="Maximum remote setup workers. 0 uses profile/default (usually 3).")
    ap.add_argument("--max-parallel-uploads", type=int, default=0, help="Maximum concurrent suite uploads. 0 uses profile/default (usually 1).")
    ap.add_argument("--powercalc-workers", type=int, default=0, help="Maximum concurrent power_calculations jobs. 0 uses profile/default (usually 1).")
    ap.add_argument("--remote-validation-images", default="")
    ap.add_argument("--remote-validation-max-images", type=int, default=0)
    ap.add_argument("--remote-validation-reference-mode", default="auto")
    ap.add_argument("--remote-mini-coco-ap50", action="store_true")
    ap.add_argument("--remote-mini-classification-eval", action="store_true")
    ap.add_argument("--remote-benchmark-task", default="auto")
    # Native producer fastpath stage: strict supported-only execution mode after generic benchmarking.
    ap.add_argument("--native-producer", action="store_true", help="Run native producer fastpaths as an EvalRun final stage and collect results into reports.")
    ap.add_argument("--native-producer-backend", action="append", default=[], help="Native producer backend to run: hailo8, hailo10h, deepx. Repeatable.")
    ap.add_argument("--native-producer-case-map", default="", help="JSON mapping model_id -> list of case ids, for example: {\"resnet50\":[\"b082\"]}.")
    ap.add_argument("--native-producer-case-policy", default="all_accepted", choices=["all_accepted", "case_map_only", "preferred_then_backfill"], help="Native producer case selection. all_accepted tries every generated b*/ case and reports unsupported cases; case_map_only only runs --native-producer-case-map; preferred_then_backfill keeps preferred map metadata but attempts all generated cases.")
    ap.add_argument("--native-producer-remote-root", default="/home/nx/native_fifo_evalsets", help="Remote root used to stage BenchmarkSets for native producer execution.")
    ap.add_argument("--native-producer-remote-tool-dir", default="/home/nx/ONNX-Splitpoint-Tool", help="Tool directory on native producer remote hosts.")
    ap.add_argument("--native-producer-hailo8-ssh", default="", help="SSH target for Hailo8 native producer, e.g. nx@192.168.0.104")
    ap.add_argument("--native-producer-hailo10-ssh", default="", help="SSH target for Hailo10H native producer, e.g. nx@192.168.0.145")
    ap.add_argument("--native-producer-deepx-ssh", default="", help="SSH target for DeepX native producer, e.g. nx@192.168.0.102")
    ap.add_argument("--native-producer-hailo8-env", default="", help="Shell prefix on Hailo8 host before native producer command.")
    ap.add_argument("--native-producer-hailo10-env", default="export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate", help="Shell prefix on Hailo10 host before native producer command.")
    ap.add_argument("--native-producer-deepx-env", default="source ~/venvs/deepx-runtime/bin/activate", help="Shell prefix on DeepX host before native producer command.")
    ap.add_argument("--native-producer-precision", default="uint8_cast_fp16")
    ap.add_argument("--native-producer-frames", type=int, default=0, help="Measured native producer frames. 0 uses profile/default 1000.")
    ap.add_argument("--native-producer-warmup", type=int, default=0, help="Native producer warmup frames. 0 uses profile/default 100.")
    ap.add_argument("--native-producer-queue-depth", type=int, default=0, help="Native FIFO queue depth. 0 uses default 3.")
    ap.add_argument("--native-producer-inflight", type=int, default=0, help="Hailo10 async inflight jobs. 0 uses default 8.")
    ap.add_argument("--native-producer-hailo-format", default="uint8")
    ap.add_argument("--native-producer-dump-outputs", action="store_true", help="Dump native producer outputs during EvalRun native stage. Off for final performance unless validating outputs.")
    ap.add_argument("--native-producer-no-copy", action="store_true", help="Do not rsync BenchmarkSets to native producer hosts; assume they already exist at remote-root/run-id.")
    ap.add_argument("--native-producer-no-build-missing-engines", action="store_true", help="Do not build missing native TensorRT engines during native producer execution.")
    ap.add_argument("--native-producer-full-baselines", action="store_true", help="Also run native full-model baselines for selected producer backends (non-FIFO execution mode).")
    ap.add_argument("--native-producer-energy", dest="native_producer_energy_enabled", action="store_true", help="Create/run native producer u.RECS energy plan after native producer stage.")
    ap.add_argument("--native-producer-energy-mode", default="", choices=["", "plan", "measure"], help="Native producer energy mode. plan creates command files; measure executes u.RECS windows immediately.")
    ap.add_argument("--native-producer-energy-frames", type=int, default=0, help="Native producer energy measured frames; 0 uses native producer frames/default.")
    ap.add_argument("--native-producer-energy-warmup", type=int, default=0, help="Native producer energy warmup frames; 0 uses native producer warmup/default.")
    ap.add_argument("--native-producer-energy-timeout", type=int, default=0, help="Native producer energy timeout per row/window. 0 uses default 900s.")
    probe = ap.add_mutually_exclusive_group()
    probe.add_argument("--window-method-validation-probe", dest="window_method_validation_probe_enabled", action="store_true", help="Run the separate screening-only marker-v2 versus historical-window probe on one successful native command.")
    probe.add_argument("--no-window-method-validation-probe", dest="window_method_validation_probe_enabled", action="store_false", help="Disable the separate window-method screening probe.")
    ap.set_defaults(window_method_validation_probe_enabled=None)
    ap.add_argument("--window-method-validation-probe-repeats", type=int, default=0, help="Independent raw traces for the screening probe. 0 uses the resolved default (3); values below 3 are marked non-decision-capable.")
    probe_raw = ap.add_mutually_exclusive_group()
    probe_raw.add_argument("--window-method-validation-probe-include-raw-parquet", dest="window_method_validation_probe_include_raw_parquet", action="store_true", help="Require and include every probe raw Parquet trace in the Debug Pack (default).")
    probe_raw.add_argument("--no-window-method-validation-probe-include-raw-parquet", dest="window_method_validation_probe_include_raw_parquet", action="store_false", help="Keep probe trace hashes in the run, but do not embed raw Parquet in the Debug Pack.")
    ap.set_defaults(window_method_validation_probe_include_raw_parquet=None)
    probe_strict = ap.add_mutually_exclusive_group()
    probe_strict.add_argument("--window-method-validation-probe-strict", dest="window_method_validation_probe_strict", action="store_true", help="Make a requested probe with zero/incomplete A/B measurements fail the workflow/CLI (default).")
    probe_strict.add_argument("--no-window-method-validation-probe-strict", dest="window_method_validation_probe_strict", action="store_false", help="Record a blocked/incomplete probe without failing the whole workflow.")
    ap.set_defaults(window_method_validation_probe_strict=None)
    ap.add_argument("--energy", action="store_true", help="Enable u.RECS energy measurement for Evaluation Workflow remote dispatches")
    ap.add_argument("--energy-scope", default="row_variant", choices=["row_variant", "dispatch"], help="Energy measurement scope for evaluation remote dispatches")
    ap.add_argument("--energy-repeat-override", type=int, default=0, help="u.RECS repeats per energy target. 0 uses remote benchmark repeats")
    ap.add_argument("--energy-phase", action="append", default=[], help="Energy phase to measure (latency, streaming). Repeatable; empty means both")
    ap.add_argument("--energy-target-policy", default="", choices=["", "all", "canonical_only", "deepx_only", "best_valid_only", "best_plus_predicted", "manual"], help="Which evaluation run profiles get u.RECS energy windows. Use 'all' with --energy-max-targets-per-run-id 0 for final all-split Energy coverage.")
    ap.add_argument("--energy-skip-backend", action="append", default=[], help="Run-id/backend to skip for energy, e.g. ort_cpu or ort_cuda. Repeatable.")
    ap.add_argument("--energy-include-run-id", action="append", default=[], help="Manual allow-list run id for energy when target_policy=manual. Repeatable.")
    ap.add_argument("--energy-exclude-run-id", action="append", default=[], help="Manual deny-list run id for energy. Repeatable.")
    ap.add_argument("--energy-heartbeat-s", type=int, default=60, help="Heartbeat/progress interval in seconds during long u.RECS energy windows")
    ap.add_argument("--energy-max-targets-per-run-id", type=int, default=None, help="Cap row-level energy targets per run-id; 0 means all. Profile default is 2 for composed split runs.")
    ap.add_argument("--energy-max-work-units-per-window", type=int, default=0, help="Hard cap on u.RECS work units per energy window; 0 disables.")
    ap.add_argument("--energy-max-window-duration-s", type=int, default=0, help="Approximate cap on active benchmark duration per energy window; 0 disables.")
    ap.add_argument("--energy-timeout-s-per-window", type=int, default=0, help="Collector timeout cap per u.RECS window; 0 uses automatic timeout.")
    ap.add_argument("--energy-sizing-probe-max-work-units", type=int, default=256, help="Max work units for the bounded sizing pilot before scaling to Min active s.")
    ap.add_argument("--energy-strict", action="store_true", help="Fail evaluation energy measurement instead of falling back/skipping when u.RECS setup is unavailable")
    ap.add_argument("--include-raw-energy-parquet", action="store_true", help="Include raw u.RECS parquet files in debug packs (large; off by default)")
    ap.add_argument("--no-model-hash", action="store_true", help="Avoid hashing large ONNX model files")
    ap.add_argument("--stop-after", choices=ALL_STAGES, default=None, help="Stop immediately after a workflow stage")
    ap.add_argument("--force-stage", action="append", default=[], help="Force a stage even when --resume would reuse it")
    ap.add_argument("--json", action="store_true", help="Print machine-readable result JSON")
    return ap


def _remote_host_json_from_args(ns: argparse.Namespace) -> str:
    raw = str(getattr(ns, "remote_host_json", "") or "").strip()
    if raw:
        return raw
    host = str(getattr(ns, "remote_host", "") or "").strip()
    if not host:
        return ""
    user = str(getattr(ns, "remote_user", "") or "").strip()
    if "@" in host and not user:
        user, host = host.split("@", 1)
    payload = {
        "id": str(getattr(ns, "remote_host_id", "") or f"{user + '@' if user else ''}{host}:{int(getattr(ns, 'remote_port', 22) or 22)}"),
        "label": str(getattr(ns, "remote_host_id", "") or host),
        "host": host,
        "user": user,
        "port": int(getattr(ns, "remote_port", 22) or 22),
        "remote_base_dir": str(getattr(ns, "remote_base_dir", "~/splitpoint_runs") or "~/splitpoint_runs"),
        "ssh_extra_args": str(getattr(ns, "remote_ssh_extra_args", "") or ""),
    }
    return json.dumps(payload, ensure_ascii=False)


_PROFILE_DRIVEN_ALLOWED_OPTIONS = {
    "--profile",
    "--out",
    "--models-root",
    "--only-model",
    "--execution-mode",
    "--run-id",
    "--profile-driven",
    "--require-run-mode",
    "--require-fresh-run",
    "--confirm-force-build",
    "--json",
}


def _validate_profile_driven_args(argv: Sequence[str]) -> None:
    supplied = {
        token.split("=", 1)[0]
        for token in argv
        if str(token).startswith("--")
    }
    unsupported = sorted(supplied - _PROFILE_DRIVEN_ALLOWED_OPTIONS)
    if unsupported:
        raise ValueError(
            "--profile-driven accepts only explicit start controls; unsupported "
            "options: " + ", ".join(unsupported)
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    ns = build_parser().parse_args(raw_argv)
    exec_mode = str(ns.execution_mode or "").strip() or ("contracts_only" if bool(ns.skip_benchmarks or ns.dry_run) else "generate_benchmarksets")
    hailo_mode = str(ns.hailo_build_mode or "reuse_only").strip().lower().replace("-", "_")
    if hailo_mode == "reuse_build_missing":
        hailo_mode = "reuse_and_build_missing"
    hailo_targets = [x.strip() for x in str(getattr(ns, "hailo_targets", "") or "").replace(";", ",").split(",") if x.strip()]
    if bool(ns.profile_driven):
        try:
            _validate_profile_driven_args(raw_argv)
            from .profile_options import (
                load_runtime_profile_snapshot,
                workflow_options_from_profile_snapshot,
            )

            _resolved_profile, start_snapshot = load_runtime_profile_snapshot(
                str(ns.profile)
            )
            supplied_options = {
                token.split("=", 1)[0]
                for token in raw_argv
                if str(token).startswith("--")
            }
            opts = workflow_options_from_profile_snapshot(
                profile_request=str(ns.profile),
                out_root=str(Path(ns.out).expanduser()),
                start_snapshot=start_snapshot,
                models_root=str(ns.models_root or ""),
                only_model_override=(
                    str(ns.only_model or "") or None
                    if "--only-model" in supplied_options
                    else None
                ),
                execution_mode_override=str(ns.execution_mode or ""),
                run_id_override=(
                    str(ns.run_id or "")
                    if "--run-id" in supplied_options
                    else None
                ),
                required_run_mode=str(ns.require_run_mode or ""),
                require_fresh_run=bool(ns.require_fresh_run),
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "status": "failed",
                        "technical_status": "failed",
                        "quality_decision": "not_evaluated",
                        "scientific_status": "not_evaluated",
                        "error": f"{type(exc).__name__}: {exc}",
                        "logs": [],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 1
    else:
        opts = WorkflowOptions(
            profile=str(ns.profile),
            out=str(Path(ns.out).expanduser()),
            models_root=str(ns.models_root or ""),
            resume=bool(ns.resume),
            dry_run=bool(ns.dry_run),
            stop_after=str(ns.stop_after or "") or None,
            only_model=str(ns.only_model or "") or None,
            include_reserve=bool(ns.include_reserve),
            skip_benchmarks=bool(ns.skip_benchmarks),
            no_remote=bool(ns.no_remote),
            run_id=str(ns.run_id or "") or None,
            max_models=ns.max_models,
            force_stage=list(ns.force_stage or []),
            no_model_hash=bool(ns.no_model_hash),
            skip_analysis=bool(ns.skip_analysis),
            execution_mode=exec_mode,
            benchmark_sets_root=str(ns.benchmark_sets_root or ""),
            benchmark_results_root=str(ns.benchmark_results_root or ""),
            result_sources=list(ns.result_source or []),
            benchmark_provider=str(ns.benchmark_provider or ""),
            benchmark_warmup=int(ns.benchmark_warmup or 1),
            benchmark_runs=int(ns.benchmark_runs or 3),
            benchmark_timeout_s=int(ns.benchmark_timeout_s or 0),
            benchmark_execution_backend=str(ns.benchmark_execution_backend or "auto"),
            benchmark_preset=str(ns.benchmark_preset or "auto"),
            benchmark_image=str(ns.benchmark_image or "default"),
            benchmark_extra_args=list(ns.benchmark_extra_arg or []),
            hailo_build_mode=hailo_mode,
            hailo_hw_arch=str(ns.hailo_hw_arch or (hailo_targets[0] if hailo_targets else "hailo8") or "hailo8"),
            hailo_build_targets=hailo_targets or [str(ns.hailo_hw_arch or "hailo8")],
            hailo_build_backend=str(ns.hailo_build_backend or "auto"),
            hailo_build_timeout_s=max(0, int(ns.hailo_build_timeout_s or 0)),
            hailo_build_full=not bool(ns.no_hailo_build_full),
            hailo_build_part1=not bool(ns.no_hailo_build_part1),
            hailo_build_part2=not bool(ns.no_hailo_build_part2),
            hailo_preset=str(ns.hailo_preset or "quick"),
            hailo_optimization_level=max(0, int(ns.hailo_opt_level or 0)),
            hailo_calib_dir=str(ns.hailo_calib_dir or ""),
            hailo_calib_count=max(0, int(ns.hailo_calib_count or 0)),
            hailo_calib_batch_size=max(1, int(ns.hailo_calib_batch_size or 1)),
            hailo_force_build=bool(ns.hailo_force_build),
            hailo_keep_artifacts=bool(ns.hailo_keep_artifacts),
            validation_mode=str(ns.validation_mode or "summary_only"),
            validation_max_cases=max(0, int(ns.validation_max_cases or 0)),
            validation_require_explicit=bool(ns.validation_require_explicit),
            validation_require_task_metrics=bool(ns.validation_require_task_metrics),
            validation_max_abs_error_threshold=float(ns.validation_max_abs_error_threshold),
            validation_mean_abs_error_threshold=float(ns.validation_mean_abs_error_threshold),
            validation_cosine_threshold=float(ns.validation_cosine_threshold),
            hardware_smoke_mode=str(ns.hardware_smoke_mode or "summary_only"),
            hardware_smoke_timeout_s=max(0, int(ns.hardware_smoke_timeout_s or 0)),
            hardware_setups_file=str(ns.hardware_setups_file or ""),
            hardware_setup_ids=list(ns.hardware_setup or []),
            hardware_group_ids=list(ns.hardware_group or []),
            rerun_generated_only=bool(getattr(ns, "rerun_generated", False)),
            native_producer_enabled=bool(getattr(ns, "native_producer", False)),
            native_producer_backends=list(getattr(ns, "native_producer_backend", []) or []),
            native_producer_model_case_map=str(getattr(ns, "native_producer_case_map", "") or ""),
            native_producer_case_policy=str(getattr(ns, "native_producer_case_policy", "") or "all_accepted"),
            native_producer_remote_root=str(getattr(ns, "native_producer_remote_root", "") or "/home/nx/native_fifo_evalsets"),
            native_producer_remote_tool_dir=str(getattr(ns, "native_producer_remote_tool_dir", "") or "/home/nx/ONNX-Splitpoint-Tool"),
            native_producer_hailo8_ssh=str(getattr(ns, "native_producer_hailo8_ssh", "") or ""),
            native_producer_hailo10_ssh=str(getattr(ns, "native_producer_hailo10_ssh", "") or ""),
            native_producer_deepx_ssh=str(getattr(ns, "native_producer_deepx_ssh", "") or ""),
            native_producer_hailo8_env=str(getattr(ns, "native_producer_hailo8_env", "") or ""),
            native_producer_hailo10_env=str(getattr(ns, "native_producer_hailo10_env", "") or ""),
            native_producer_deepx_env=str(getattr(ns, "native_producer_deepx_env", "") or ""),
            native_producer_precision=str(getattr(ns, "native_producer_precision", "") or "uint8_cast_fp16"),
            native_producer_frames=max(0, int(getattr(ns, "native_producer_frames", 0) or 0)),
            native_producer_warmup=max(0, int(getattr(ns, "native_producer_warmup", 0) or 0)),
            native_producer_queue_depth=max(0, int(getattr(ns, "native_producer_queue_depth", 0) or 0)),
            native_producer_inflight=max(0, int(getattr(ns, "native_producer_inflight", 0) or 0)),
            native_producer_hailo_format=str(getattr(ns, "native_producer_hailo_format", "") or "uint8"),
            native_producer_dump_outputs=bool(getattr(ns, "native_producer_dump_outputs", False)),
            native_producer_no_copy=bool(getattr(ns, "native_producer_no_copy", False)),
            native_producer_no_build_missing_engines=bool(getattr(ns, "native_producer_no_build_missing_engines", False)),
            native_producer_energy_enabled=bool(getattr(ns, "native_producer_energy_enabled", False)),
            native_producer_energy_mode=str(getattr(ns, "native_producer_energy_mode", "") or "plan"),
            native_producer_energy_frames=max(0, int(getattr(ns, "native_producer_energy_frames", 0) or 0)),
            native_producer_energy_warmup=max(0, int(getattr(ns, "native_producer_energy_warmup", 0) or 0)),
            native_producer_energy_timeout_s=max(0, int(getattr(ns, "native_producer_energy_timeout", 0) or 0)),
            window_method_validation_probe_enabled=getattr(ns, "window_method_validation_probe_enabled", None),
            window_method_validation_probe_repeats=max(0, int(getattr(ns, "window_method_validation_probe_repeats", 0) or 0)),
            window_method_validation_probe_include_raw_parquet=getattr(ns, "window_method_validation_probe_include_raw_parquet", None),
            window_method_validation_probe_strict=getattr(ns, "window_method_validation_probe_strict", None),
            remote_host_json=_remote_host_json_from_args(ns),
            remote_host_id=str(ns.remote_host_id or ""),
            remote_hosts_file=str(ns.remote_hosts_file or ""),
            remote_host=str(ns.remote_host or ""),
            remote_user=str(ns.remote_user or ""),
            remote_port=int(ns.remote_port or 22),
            remote_base_dir=str(ns.remote_base_dir or "~/splitpoint_runs"),
            remote_ssh_extra_args=str(ns.remote_ssh_extra_args or ""),
            remote_working_dir=str(ns.remote_working_dir or ""),
            remote_provider=str(ns.remote_provider or "auto"),
            remote_venv=str(ns.remote_venv or ""),
            remote_transfer_mode=str(ns.remote_transfer_mode or "bundle"),
            remote_reuse_bundle=not bool(ns.no_remote_reuse_bundle),
            remote_no_reuse_bundle=bool(ns.no_remote_reuse_bundle),
            remote_resume=not bool(ns.no_remote_resume),
            remote_no_resume=bool(ns.no_remote_resume),
            remote_repeats=max(1, int(ns.remote_repeats or 1)),
            remote_warmup=max(0, int(ns.remote_warmup or 0)),
            remote_iters=max(0, int(ns.remote_iters or 0)),
            remote_timeout_s=max(0, int(ns.remote_timeout_s or 0)),
            remote_add_args=" ".join(str(x) for x in list(ns.remote_add_arg or []) if str(x).strip()),
            remote_throughput_frames=max(0, int(ns.remote_throughput_frames or 0)),
            remote_throughput_warmup_frames=max(0, int(ns.remote_throughput_warmup_frames or 0)),
            remote_throughput_queue_depth=max(1, int(ns.remote_throughput_queue_depth or 1)),
            remote_validation_images=str(ns.remote_validation_images or ""),
            remote_validation_max_images=max(0, int(ns.remote_validation_max_images or 0)),
            remote_validation_reference_mode=str(ns.remote_validation_reference_mode or "auto"),
            remote_mini_coco_ap50=bool(ns.remote_mini_coco_ap50),
            remote_mini_classification_eval=bool(ns.remote_mini_classification_eval),
            remote_benchmark_task=str(ns.remote_benchmark_task or "auto"),
            energy_enabled=bool(ns.energy),
            energy_scope=str(ns.energy_scope or "row_variant"),
            energy_repeat_override=max(0, int(ns.energy_repeat_override or 0)),
            energy_phases=list(ns.energy_phase or []),
            energy_target_policy=str(ns.energy_target_policy or ""),
            energy_skip_backends=list(ns.energy_skip_backend or []),
            energy_include_run_ids=list(ns.energy_include_run_id or []),
            energy_exclude_run_ids=list(ns.energy_exclude_run_id or []),
            energy_heartbeat_s=max(10, int(ns.energy_heartbeat_s or 60)),
            energy_strict=bool(ns.energy_strict),
            energy_max_targets_per_run_id=int(ns.energy_max_targets_per_run_id) if ns.energy_max_targets_per_run_id is not None else -1,
            energy_max_work_units_per_window=max(0, int(ns.energy_max_work_units_per_window or 0)),
            energy_max_window_duration_s=max(0, int(ns.energy_max_window_duration_s or 0)),
            energy_timeout_s_per_window=max(0, int(ns.energy_timeout_s_per_window or 0)),
            energy_sizing_probe_max_work_units=max(1, int(ns.energy_sizing_probe_max_work_units or 256)),
            energy_include_raw_parquet_in_debug_pack=bool(ns.include_raw_energy_parquet),
            required_run_mode=str(ns.require_run_mode or ""),
            require_fresh_run=bool(ns.require_fresh_run),
        )
    opts.force_build_confirmed_backends = tuple(dict.fromkeys(
        list(ns.confirm_force_build) + (["hailo"] if ns.hailo_force_build else [])
    ))
    opts.force_build_confirmation_source = "cli_explicit_option"
    if bool(getattr(ns, "rerun_generated", False)):
        opts.resume = True
        for _st in ["run_benchmarks", "validate_outputs", "hardware_smoke", "aggregate_results", "generate_report"]:
            if _st not in opts.force_stage:
                opts.force_stage.append(_st)
        opts.skip_benchmarks = False
        opts.execution_mode = "generate_and_run"
        opts.remote_reuse_bundle = False
        opts.remote_no_reuse_bundle = True
        opts.remote_resume = False
        opts.remote_no_resume = True
    if bool(getattr(ns, "finalize_partial", False)):
        opts.resume = True
        opts.skip_benchmarks = True
        opts.no_remote = True
        opts.energy_enabled = False
        # Recompute summaries and reports from whatever model artifacts exist.
        # Do not force run_benchmarks; this is explicitly a partial-run finalizer.
        for _st in ["validate_outputs", "hardware_smoke", "aggregate_results", "generate_report"]:
            if _st not in opts.force_stage:
                opts.force_stage.append(_st)
        if not opts.run_id and str(getattr(ns, "run_id", "") or "").strip():
            opts.run_id = str(ns.run_id).strip()

    logs: list[str] = []
    def _log(msg: str) -> None:
        logs.append(msg)
        if not bool(ns.json):
            print(msg, flush=True)
    try:
        result = EvaluationWorkflowRunner(opts, log=_log).run()
        payload = result.to_dict()
        payload["logs"] = logs
    except WorkflowRunControlError as exc:
        payload = {
            "ok": False,
            "status": str(exc.status),
            "technical_status": str(exc.status),
            "quality_decision": "not_evaluated",
            "scientific_status": "not_evaluated",
            "error_code": str(exc.error_code),
            "error": str(exc),
            "owner": dict(exc.owner),
            "logs": logs,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 130 if str(exc.status).strip().lower() == "cancelled" else 2
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "failed",
            "technical_status": "failed",
            "quality_decision": "not_evaluated",
            "scientific_status": "not_evaluated",
            "error": f"{type(exc).__name__}: {exc}",
            "error_code": str(getattr(exc, "error_code", "") or ""),
            "logs": logs,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 1
    if ns.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    else:
        print(f"[done] {(payload.get('completion') or {}).get('message') or result.status}: {result.run_dir}", flush=True)
        print(
            "[decision] "
            f"technical_status={getattr(result, 'technical_status', result.status)} "
            f"quality_decision={getattr(result, 'quality_decision', 'not_evaluated')} "
            f"scientific_status={getattr(result, 'scientific_status', 'not_evaluated')}",
            flush=True,
        )
        print(f"manifest: {result.manifest_path}", flush=True)
    if str(getattr(result, "status", "") or "").strip().lower() == "cancelled":
        return 130
    return 0 if bool(getattr(result, "ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
