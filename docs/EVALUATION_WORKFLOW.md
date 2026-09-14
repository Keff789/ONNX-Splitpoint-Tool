# Evaluation Workflow

The Evaluation Workflow is the reproducible campaign layer around the existing BenchmarkSet infrastructure.

```text
Evaluation Profile auswählen
→ Start Evaluation Workflow
→ Jobs beobachten
→ Results Bundle / Dashboard öffnen
```

## Architecture

```text
EvaluationWorkflowRunner
  - resolves profile and models
  - writes EvaluationRuns/<run_id>/
  - creates formal stage artifacts and resume decisions
  - exposes a parent/subjob tree in the Jobs tab
  - delegates real BenchmarkSet work to the existing BenchmarkSet pipeline
  - normalizes benchmark/validation/hardware results
  - generates dashboard, thesis metrics, LaTeX tables and figures

Existing BenchmarkSet pipeline
  - split export
  - Hailo HEF build/reuse
  - YOLO raw-head / host-tail policies
  - benchmark_suite.py
  - local/remote benchmark execution
  - validation reports
```

The workflow is deliberately an orchestration/reporting layer. It should not reimplement the heavy BenchmarkSet generator.

## 2.75.49 YOLOv7 DeepX-Full closeout order

The model-bound decoder changes postprocessing inside the measured evaluation
path. Therefore the order is fixed:

1. retain the accepted CPU-only decoder A/B evidence for the pinned YOLOv7
   model and B500 cohort (`status=completed`, `acceptance.status=accepted`,
   `official_coco.*.status=ok`); the narrow DeepX repair does not rerun it;
2. start the packaged v2.75.49 YOLOv7-only profile as a fresh GUI workflow,
   never Resume the v2.75.48 run;
3. verify one model/one case, 9/9 Native rows and 27/27 independent Native
   repetitions before opening the next campaign phase;
4. run the later 20-candidate score-independent audit Native-first with Energy
   disabled;
5. measure Energy only after the winner/selection rule is frozen.

A failed or unavailable official COCO probe blocks the GUI anchor. The user
must not replace it with a relative candidate-versus-reference pass produced
by the same decoder. The technical anchor uses paired internal B500 Quality;
Official COCO is mandatory again for the later final Detection run.

## CLI

```bash
python -m onnx_splitpoint_tool.workflow.run_evaluation \
  --profile smoke_regression_v1 \
  --models-root "$HOME/Models" \
  --out "$HOME/Models/EvaluationRuns" \
  --execution-mode generate_and_run
```

Useful flags:

```text
--resume
--only-model <model-id>
--max-models <n>
--dry-run
--stop-after <stage>
--skip-benchmarks
--no-remote
--json
```

Wrapper:

```bash
python scripts/run_evaluation_workflow.py \
  --profile smoke_regression_v1 \
  --out "$HOME/Models/EvaluationRuns"
```

## Stage order

```text
resolve_profile
resolve_model
check_validation_assets
prepare_model
analyze_model
select_split_candidates
prepare_full_baselines
generate_benchmark_set
build_backend_artifacts
run_benchmarks
validate_outputs
hardware_smoke
aggregate_results
generate_report
```

## Run directory

```text
EvaluationRuns/<profile>_<timestamp>/
  profile.yaml
  profile_resolution.json
  run_manifest.json
  artifact_index.json
  evaluation_workflow.log
  jobs/
    job_plan.json
    job_summary.json
    job_events.jsonl
    job_timeline.md
    resume_summary.json
  stages/
    resolve_profile/stage_result.json
    aggregate_results/stage_result.json
    generate_report/stage_result.json
    run_native_producers/
      native_performance/stage_result.json
      native_performance/stage_snapshot.json
      native_coordinator/stage_result.json
  models/<model_id>/
    model_manifest.json
    stages/<stage>/stage_result.json
    stages/<stage>/resume_decision.json
    analysis/
      analysis.json
      candidate_ranking.json
      prediction.json
      final_candidate_plan.json
    full_baselines/
      full_baseline_plan.json
      output_contracts.json
    benchmark_set/
      benchmark_set.json
      benchmark_plan.json
      generator_binding.json
      generation_decisions.json
      legacy_suite/
    benchmark_results/
      normalized_results.json
      result_ingestion_manifest.json
      remote_benchmark_status.json
    validation/
      validation_summary.json
      validation_case_matrix.csv
    hardware/
      hardware_smoke_status.json
  reports/
    native_energy_measurements/
      stages/native_energy/stage_result.json
      checkpoints/native_energy/journal.json
      checkpoints/native_energy/rows/*.json
    summary.csv
    model_summary.csv
    hardware_summary.csv
    prediction_vs_benchmark.csv
    result_dashboard.md
    result_dashboard.json
    thesis_metrics.csv
    backend_speedups.csv
    tables/*.tex
    figures/*.png
```

## Status semantics

- `ok`: required artifacts and measurements are present.
- `warn`: non-blocking warning; later benchmark/validation/hardware stages may still be valid.
- `partial`: missing required work or incomplete measurement/validation evidence.
- `failed`: blocking failure.
- `skipped`: intentionally skipped by profile/options.
- `cancelled`: execution began but did not reach a terminal decision;
  `complete=false` is mandatory.

Lifecycle state is recorded separately as `not_started`, `running`,
`completed`, `failed`, or `cancelled`. A failed stage can be decision-complete;
a running or cancelled stage cannot.

## Atomic Native checkpoints

The Native coordinator commits Performance before it starts downstream
validation, probes, or Energy. In the current Standard/Final campaign this
requires exactly 63 imported rows, three physical producer setups with 21 rows
each, the same three models on every setup, five Split rows and two Full rows
per model, and no missing, duplicate, unexpected, or conflicting identity.
The checkpoint hashes its frozen stage snapshot and imported report artifacts.

Native Energy materializes the full plan as individual atomic row files before
preflight can start a measurement. Row files are authoritative; `journal.json`
is a derived index. On cancellation, only the single active row may become
`cancelled`; all later rows remain `not_started`. Managed Resume validates the
same plan and invocation hashes, imports a verified result left by a finished
child if necessary, and otherwise continues at the first nonterminal row. A
terminal Native Performance checkpoint is reused and is not executed again.

Plan membership is technical and independent of Quality, Semantics, pairing,
or claim status:

```text
runtime_success
AND energy_command_preflight_ok
AND (full_baseline OR split_has_valid_part2_input)
```

The planner archives this decision on every selected row. Split input validity
comes from the portable, hash-bound TensorRT Part-2 metadata; Full baselines do
not require a Split input. Quality and pairing can only downgrade the result to
a non-claimable Runtime Observation. They cannot remove a physically executable
measurement row in the direct, non-variant path. A model-local Native preflight
failure removes only that row; independent valid rows may still transfer and
start Performance, Native and Energy work. A physically completed row with a
technical Quality failure retains raw energy with
`energy_quality_qualified=false`,
`energy_quality_status=raw_energy_quality_not_qualified` and
`native_energy_after_technical_error=collect_raw_quality_unqualified`.
Unstartable rows receive no synthetic energy and expose a concrete not-started
reason.

Generic Runner Energy is enabled only when it is explicitly included by the
resolved profile. Such intent is preserved as `native_and_generic` when the
combined path is supported. An inconsistent or unsupported Generic-Energy
request becomes a visible planning blocker and is not silently reduced to
`native_only`; an explicitly Native-only request remains `native_only`.
Infrastructure failures such as unavailable u.RECS/Jetson SSH,
authentication, platform locking or collector initialization remain global.
This isolation contract applies to the direct, non-variant path. The variant
coordinator is unchanged and is not claimed by this closure.

Complete split measurements are only counted when a composed latency or part1+part2(+transfer) latency is present. Full-baseline latency and part2-only/host-tail-only rows are never treated as complete split latency.

## v51 Hailo full-runtime handling

v51 keeps the existing BenchmarkSet generator as the source of truth for Full-Hailo/Raw-Head HEF generation, but makes the runtime side stricter and clearer:

```text
- full baseline rows are normalized as variant=full
- mixed rows such as hailo8_to_tensorrt no longer become best_full_backend
- Hailo runtime evidence is reported separately from model health
- remote execution searches common Hailo Python environments and exposes their site-packages via SPLITPOINT_EXTRA_SITES
```

If `/dev/hailo0` and `hailort_service` are present but Python cannot import `hailort`/`hailo_platform`, the workflow now reports that as missing Hailo runtime binding rather than silently treating Hailo as measured.

## v51c Hailo DFC provisioning

The managed Hailo DFC installer is now kept in the **Hardware** tab only, so the Evaluation Workflow tab stays focused on profile → start → jobs → results.

```text
Hardware → DFC env status
Hardware → Open wheels
Hardware → Install/Repair DFC
```

The compiler wheels must be placed under:

```text
onnx_splitpoint_tool/resources/hailo/hailo8/*.whl
onnx_splitpoint_tool/resources/hailo/hailo10/*.whl
```

The provisioning helper creates/repairs:

```text
~/.onnx_splitpoint_tool/hailo/venv_hailo8/
~/.onnx_splitpoint_tool/hailo/venv_hailo10/
```

v51c fixes two installer problems:

```text
- Missing Graphviz development headers are a warning, not an early fatal stop.
  The installer attempts the DFC wheel first and reports the apt command only if pip fails on pygraphviz.
- If a wheel is tagged for a specific Python, e.g. cp310, the installer auto-selects a matching Python executable when available.
```

This is the local/WSL **DFC compiler** environment for ONNX→HEF builds. The remote HailoRT runtime venv, for example `~/hailo_py/bin/activate`, remains configured through the central Remote Host / Evaluation Profile settings.
