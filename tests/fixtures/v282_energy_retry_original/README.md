# Original native energy retry evidence, 2026-09-13

These files are byte-preserving copies from the user's
`completsetdev_20260913_123411` debug pack. They are **not synthetic**.

- `yolo26m/checkpoint.json`: original row 0067, Hailo8→TensorRT, b038.
- `yolo26s/checkpoint.json`: original row 0068, Hailo8→TensorRT, b021.
- Each `measurement/` contains its original `energy_aggregate.json`, all three
  initial stdout logs and the selected repeat-0/attempt-1 stdout log.
- Original absolute paths, failed decisions, selection histories, completion
  hashes and checkpoint seals remain unchanged. Original file timestamps were
  retained on fixture import, but replay's start-time contract does not invent
  an execution date; it uses the saved checkpoint timestamp.

Replay maps each checkpoint's exact `execution.output_dir` to its sibling
`measurement/` directory only in memory. This explicit archive-root projection
is reported by the importer. Product imports use their real bound paths.

The tests replay the complete checkpoint → aggregate → selection → completion
contract and demonstrate the former initial-stdout hash mismatch. The original
failed attempts remain preserved; effective statistical n remains 3. No device
execution, raw-trace recalculation, quality admission or scientific claim is
inferred from this successful import.

Excluded: models, tensors, calibration images, CUDA packages and raw arrays.
The selected stdout and aggregate contain the recorded binding evidence used
by the importer; the rest of the original debug pack is the retained authority.

Source-release ZIPs normalize filesystem mtimes to 1980. Archive replay therefore
uses the original `energy_aggregate_mtime_ns` and original aggregate SHA-256
already bound inside the sealed checkpoint, and reports both the historical and
extracted timestamp. It still checks the original timestamp against the saved
execution start and still rejects changed bytes. Missing original timestamp/hash
provenance is a failed archive import; replay does not touch any source file's
mtime to manufacture freshness.
