# Native HailoRT → TensorRT FIFO Fastpath

This directory contains the first generic native fastpath for split pipelines:

`Hailo Part1 HEF → FIFO → native TensorRT Part2 engine`

It is intentionally generic: the runner is configured by file paths and tensor sizes
rather than by a YOLOv7-specific code path. It is meant as a smoke/performance path
for comparing the optimized native FIFO implementation against the generated Python
runner.

The wrapper script is:

```bash
python scripts/native_hailo_trt_fifo_from_benchmarkset.py --benchmark-set <BS> --case b066
```

The C++ runner writes a JSON report with stage timings, pipeline FPS, and queue/FIFO
statistics.
