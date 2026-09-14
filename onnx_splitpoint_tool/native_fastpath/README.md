# Native HailoRT → TensorRT FIFO fastpath

This is the first generic native C++ fastpath for measuring the HailoRT→TensorRT
pipeline without the Python/NumPy/ORT mapping loop. It consumes existing split
artifacts:

- Hailo Part1 HEF
- Native TensorRT Part2 engine
- optional queue depth / frame count

The runner is deliberately postprocess-free. It measures the native handoff and
pipeline throughput; semantic validation remains in the normal Python runner.

Current limitation: one Hailo input vstream, one Hailo output vstream, one TRT
input. This covers the YOLOv7 b066 paper-fingerprint case and many simple split
contracts. Multi-output boundaries remain on the generic Python runner until a
later fastpath revision.
