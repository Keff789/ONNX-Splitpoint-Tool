# v59ap: Native HailoRT → TensorRT FIFO fastpath

This release adds a first generic native C++ smoke runner for the HailoRT→TensorRT split path.

## Why

The Python runner already measures pipeline streaming FPS, but the YOLOv7 b066 diagnostics showed that the remaining gap is the generic Python/NumPy handoff. The TensorRT part2 engine itself is fast, and the uint8 bridge engine reduces H2D latency, but the generated Python runner still spends too much time in the P2 thread.

## What is included

- `onnx_splitpoint_tool/native_fastpath/hailo_trt_fifo_fastpath.cpp`
- `scripts/native_hailo_trt_fifo_smoke.py`

The C++ fastpath uses:

- persistent HailoRT VStreams
- persistent TensorRT engine/context
- preallocated FIFO boundary buffers
- one P1 producer thread and one P2 consumer thread
- JSON output with P1, P2, H2D/enqueue/D2H-sync and paper-equivalent FPS

Current limitation: one Hailo input vstream, one Hailo output vstream, and one TensorRT input. This covers the YOLOv7 b066 paper-fingerprint split. Multi-output split boundaries remain on the Python runner until the next native-fastpath revision.

## Example

```bash
cd ~/ONNX-Splitpoint-Tool
export BS=/home/nx/yolov7_paper_benchmark_20260625_070717

python scripts/native_hailo_trt_fifo_smoke.py \
  --benchmark-set "$BS" \
  --case b066 \
  --hw-arch hailo8 \
  --precision uint8_cast_fp16 \
  --frames 500 \
  --warmup 50 \
  --queue-depth 2
```

For an upper-bound test that does not copy TensorRT outputs back to host:

```bash
python scripts/native_hailo_trt_fifo_smoke.py \
  --benchmark-set "$BS" \
  --case b066 \
  --hw-arch hailo8 \
  --precision uint8_cast_fp16 \
  --frames 500 \
  --warmup 50 \
  --queue-depth 2 \
  --no-copy-outputs
```

## Interpreting the JSON

- `fps_makespan`: measured throughput over the consumer side.
- `paper_equivalent_cycle_ms`: `max(mean(P1), mean(P2))`.
- `paper_equivalent_fps`: `1000 / paper_equivalent_cycle_ms`.
- `p1_ms`: Hailo write+read timing.
- `p2_thread_ms`: TensorRT H2D + enqueue + optional output copy + sync.
- `trt_h2d_ms`, `trt_enqueue_ms`, `trt_d2h_sync_ms`: split of the P2 thread inside the native TensorRT wrapper.

## Important

This is a performance smoke path, not a semantic-validation path. Use the generated Python runner for semantic AP50/visual validation. The native fastpath exists to isolate the runtime/handoff overhead and make a fairer performance comparison to the old C++ paper pipeline.


## Native Detection Dual Endpoint (v2.78.3)

Detection remains one logical Native Runner.  One invocation executes two
isolated, sequential phases with fresh runtime instances and identical model,
HEF, TensorRT engine, image, prepared-input, boundary and raw-output contracts:

1. `raw_model_outputs`: the C++ HailoRT→bounded-FIFO→TensorRT fastpath.  This
   is the primary hardware-performance and Generic↔Native ranking endpoint.
2. `completed_task`: the existing correctness-first runtime including frozen
   decode, class-aware NMS, inverse letterbox, result materialisation and
   completion evidence.  This is an application-throughput endpoint.

The two values are not taken from one back-pressured consumer loop.  The
Completed-Task phase may stall the FIFO and would otherwise contaminate an
earlier timestamp.  The parent invocation therefore runs both phases separately
and publishes one combined result only after exact HEF, engine, source-image,
prepared-input, boundary and raw-output parity has been verified.  Energy remains
bound to the Completed-Task endpoint until an endpoint-specific energy campaign
is explicitly configured.

### Completion-tail instrumentation

`scripts/native_detection_completion_tail_canary.py` replays a saved raw-output
manifest through the exact frozen `DetectionCompletionRuntime` without Hailo or
TensorRT work.  It reports inclusive timings for head mapping, raw-output hashing,
contract verification, format normalisation, YOLO decode, class-aware NMS, result
canonicalisation and evidence hashing, plus a cProfile artifact.  Absolute timing
comparisons must be made on the same host CPU as the integrated Native run.
