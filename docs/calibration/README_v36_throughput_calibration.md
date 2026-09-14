# Throughput calibration profile v36

This increment enables a first benchmark-derived calibration layer for the
throughput-oriented handover metric.  The embedded profile
`yolov7_streaming_v1` was derived from the previously benchmarked YOLOv7
Hailo8↔TensorRT run.

Source data:
- `yolov7_streaming_reference.csv`

What is calibrated:
- direction-aware adjustment of the raw predicted handover time
- separate handling for `Hailo -> Other` and `Other -> Hailo`
- hinge features on cut-size excess and imbalance excess

Important limitation:
- the calibration is only applied when an imbalance estimate is available
- generic callers without richer metadata fall back to the raw interpretable
  heuristic
