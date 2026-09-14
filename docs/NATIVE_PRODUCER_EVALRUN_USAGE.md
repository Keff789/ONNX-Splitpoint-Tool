# Native Producer EvalRun Usage

The native producer fastpath is an additional strict execution mode. It does not replace the generic runner.

## What is measured

Split native producers:

- `hailo8_to_trt`: C++ HailoRT VStreams FIFO producer + native TensorRT consumer
- `hailo10h_to_trt`: HailoRT InferModel async FIFO producer + native TensorRT consumer
- `deepx_to_trt`: dx_engine FIFO producer + native TensorRT consumer

Full baselines are not FIFO pipelines because no inter-device split boundary exists. For fairness they must still use optimized native full runners where available:

- TensorRT full: native TensorRT engine
- Hailo8 full: native HailoRT full runner / existing Hailo runtime benchmark
- Hailo10 full: InferModel async throughput runner
- DeepX full: dx_engine full runner

Reports should distinguish `full_native_baseline` from `split_native_fifo`.

## Enable in a profile

```yaml
native_producers:
  enabled: true
  backends: [hailo8, hailo10h, deepx]
  case_policy: all_accepted
  precision: uint8_cast_fp16
  frames: 1000
  warmup: 100
  queue_depth: 3
  inflight: 8
  hailo_format: uint8
  remote_root: /home/nx/native_fifo_evalsets
  remote_tool_dir: /home/nx/ONNX-Splitpoint-Tool
  build_missing_engines: true
  copy_benchmarksets: true
  strict_supported_only: true
  remotes:
    hailo8:
      ssh: nx@192.168.0.104
    hailo10h:
      ssh: nx@192.168.0.145
      env: "export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate"
    deepx:
      ssh: nx@192.168.0.102
      env: "source ~/venvs/deepx-runtime/bin/activate"
```

Or patch a profile with:

```bash
python scripts/configure_native_producer_profile.py --profile profiles/YourProfile.yaml
```

## Energy

Evaluation Workflow Energy is Native-only; Generic Runner Energy is hard
disabled. Native producer Energy uses `reports/native_producer_summary.json`
and admits every runtime-successful row whose Energy command/preflight is
constructible and which is either a Full baseline or has one verified Part-2
input. Quality, Semantics, pairing, and claim eligibility are downstream
annotations only.

The managed workflow writes:

```text
reports/native_energy_measurements/stages/native_energy/stage_result.json
reports/native_energy_measurements/checkpoints/native_energy/journal.json
reports/native_energy_measurements/checkpoints/native_energy/rows/*.json
```

Every planned row exists before measurement as `not_started` and is atomically
replaced through `running` to `completed`, `failed`, or `cancelled`. At most one
row can be `running`. A pipe closure or signal may cancel only that active row;
it must not label later `not_started` rows with `BrokenPipeError`.

`--resume-checkpoint` is an internal managed-workflow contract. It validates
the exact invocation, summary, plan and row identities. Terminal rows are
reused, a verified child result can close the parent-import gap without a
second attempt, and execution resumes at the first nonterminal row. The
terminal Native Performance checkpoint under
`stages/run_native_producers/native_performance/` is reused first, so a
synthetic Energy Resume never repeats Performance. Historical `--resume-existing`
mode cannot be mixed with this journal.

These Resume rules apply only to new runs and test fixtures. They do not
authorize continuation or modification of an archived diagnostic run.
