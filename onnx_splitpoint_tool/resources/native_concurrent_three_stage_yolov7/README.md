# ONNX Splitpoint Tool – YOLOv7 Native Three-Stage Canary v1

This is a **read-only, independent hardware canary** for the exact current
v2.78.2 YOLOv7-paper `b066` artifacts. It does not install or patch the tool.

## Purpose

The canary tests one logical Native invocation with three concurrent internal
stages:

1. **P1:** image read, letterbox preprocessing and Hailo-8 part 1;
2. **P2:** boundary handoff, TensorRT part 2 and D2H of the three raw heads;
3. **Postprocessing:** exact sparse NumPy multiscale decode, frozen class-aware
   NMS, inverse letterbox and canonical detection records.

P1 and P2 execute in C++. A dedicated third C++ worker invokes the contract-bound
NumPy adapter through an in-process `ctypes` callback. TensorRT copies each raw
head directly into a pinned host ring slot; NumPy reads that slot without an
additional raw-head copy.

This remains **one Native Runner architecture**. `P1`, `P2` and
`Postprocessing` are internal pipeline stages, not separate scientific runners.

The completed endpoint is deliberately named `completed_detection`, not
`device_completed_detection`: in this universal canary the third stage runs on
the Orin NX host CPU through NumPy. A future TensorRT-integrated decode/NMS
variant would be a different implementation location under the same explicit
endpoint contract, and must not be conflated in reports.

## Quality/evidence separation

The timed callback contains no SHA-256 hashing, JSON evidence construction,
contract discovery or slow-oracle execution.

After the three performance repetitions, a separate 32-image postflight runs
outside the performance window and requires all of the following for every
image:

- exact raw-head SHA-256 values relative to the already validated multi-image
  corpus;
- exact parity between the new sparse implementation and the frozen current
  v2.78.2 quality oracle;
- equality to the previously validated oracle detections.

## Measurement contract

Default settings:

- 32 deterministic COCO-val images;
- three fresh Native runtime repetitions;
- 100 fully drained warm-up frames per repetition;
- 1,000 measured frames per repetition;
- P1/P2 FIFO depth 3;
- P2/Postprocessing ring depth 4;
- exact current `uint8_dequant_fp16` TensorRT engine;
- no HEF build, TensorRT build, B500, energy run or tool mutation.

The report keeps the following measurements separate:

- **P1:** image-read/letterbox preprocessing, Hailo-8 inference and combined stage time;
- **P2:** boundary handoff, TensorRT inference plus raw-head D2H and combined stage time;
- **Postprocessing:** fast multiscale decode, class-aware NMS, inverse letterbox and canonical records;
- observed `raw_model_outputs` throughput inside the coupled three-stage pipeline;
- stage-theoretical raw capacity from `max(mean(P1), mean(P2))`;
- observed `completed_detection` throughput;
- completed/raw ratio and both queue-wait distributions.

The distinction between observed and stage-theoretical raw throughput is deliberate:
when Postprocessing becomes slower than P1/P2, bounded queues correctly apply
backpressure to the coupled pipeline instead of hiding it.

## PASS gate

`PASS_THREE_STAGE_TARGET_MET` requires:

- 3/3 runtime repetitions complete;
- exact 32-image result parity in each measured repetition;
- exact 32/32 postflight raw-head and oracle parity;
- median raw throughput at least 90 FPS;
- median completed-detection throughput at least 90 FPS;
- median completed/raw ratio at least 0.90;
- postprocessing P95 no greater than 10 ms and worst measured frame no greater
  than 20 ms;
- unchanged HEF, engine and frozen oracle sources.

A parity-preserving but slower run is reported as
`PASS_THREE_STAGE_EXACT_SPEED_OPEN`, not silently rejected.

## Run from Smartmirror2

```bash
(
set -Eeuo pipefail

DL="$HOME/Downloads"
ZIP="$(
  find "$DL" -maxdepth 1 -type f \
    -name 'onnx_splitpoint_yolov7_three_stage_canary_v1*.zip' \
    -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"

test -n "$ZIP" || {
  echo "STOP: Three-Stage-Canary-ZIP unter $DL fehlt."
  exit 66
}

cd "$DL"
rm -rf onnx_splitpoint_yolov7_three_stage_canary_v1
unzip -q "$ZIP"
cd onnx_splitpoint_yolov7_three_stage_canary_v1

set +e
bash run_from_smartmirror2.sh
RC=$?
set -e

echo
echo "LOCAL_CANARY_WRAPPER_RC=$RC"
)
```

Upload the file printed as `RESULT_ZIP=`. Do not install the earlier v2.78.3
prototype before this canary is evaluated.
