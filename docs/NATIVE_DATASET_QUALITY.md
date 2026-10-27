# Native dataset Quality

`scripts/native_dataset_quality_remote.py` prepares and executes one explicitly
selected detection case through the ordinary SSHTransport, EvaluationRunLock,
physical device lock, resource owner observation and remote process leases.
The capture entrypoint is `scripts/native_dataset_quality.py`. It opens the
existing Hailo Native FIFO runtime once and submits each of the frozen 5000
images once. Existing preprocessing, exact artifact selection, quantization,
boundary conversion, TensorRT execution and completion remain in the existing
Native functions. No performance or energy measurements are emitted.

The config schema is `onnx-splitpoint/native-dataset-quality-config`, version 1.
It supplies a fresh evaluation identity, the untouched prior sealed Native
eight-artifact selection, an exact measured Native FPS result descriptor, the
measured completion policy, the canonical Quality contract, frozen ground truth
and image hashes, and the existing metric policy and B1000 bootstrap settings.
The prior Quality run identifiers remain inside `prior_artifact_selection`;
they are never reused as the new capture identity. Inputs and artifacts are
hash checked. Engines must exist and are opened directly; builds are forbidden.

For each image the normal completion factory binds its actual geometry and
source tensor signature. The complete processor semantics and source endpoint
are compared with the measured Native policy. Geometry, declaration attestation
and truthful implementation provenance are newly bound. Detections come from
the FIFO's own normal completion attestation, verified with the existing strict
or fast verifier. There is no additional host NMS. A raw-head model still runs
the decode/NMS already present in its timed Native completion path.

The exporter publishes `native_completion_evidence.json`,
`composed_candidate.json`, then `composed_request.json`, each into a fresh
directory. The ordinary central Quality envelope carries a fresh
`native-dataset-quality-producer`, exact candidate-record hash, immutable prior
artifact selection, actual implementation source descriptors, completed counts
and the per-image completion evidence descriptor. The old canonical CPU Quality
contract describes the unchanged reference/record compatibility; the separate
candidate completion contract and attestations describe the actual Native
execution. The historical CPU reference and its postprocessing are preserved.

`native_dataset_quality_consumer.load_native_quality_request` reads the genuine
previous normal CPU reference status with its original reference run directory,
verifies its immutable bytes through the normal reader, and loads the ordinary
paired request. It checks the fresh Native identity, current published Product
source bytes, 5000 image identities, prepared input/image bindings, every normal
completion attestation and exact equality of candidate detections with completed
detections. Numeric class IDs use the same model-label indexing as the reference;
no class-name annotation changes the attested records.
`evaluate_native_quality_request(loaded, service=...)` uses the caller's existing
ManagementQualityService and normal result enrichment. It creates no pool and
runs no reference inference or additional detector postprocessor.

For separately published Native capture source epochs, the management caller
supplies the genuine producer `SOURCEFREEZE_READY` descriptor as
`producer_source_freeze`. The normal installed-source guard verifies it. The
consumer checks unchanged runtime/statistics sources and the reviewed model
metadata difference, then checks the finite producer source files against that
exact release. A request cannot choose a replacement source root. Actual
producer and consumer identities remain separate in the shared service.

Preparation JSON supplies `config_path`, normal `target`, remote `python`,
fresh absolute `remote_root`, fresh `output_root`, and the immutable
`source_freeze` descriptor. Optional `reserve_bytes`, `forecast_bytes` and
`timeout_s` keep ordinary capacity admission explicit; the default reserve is
24 GiB and the full forecast is retained. The preparation must run from the
published source export and pass normal installed-source integrity.

```bash
python -B scripts/native_dataset_quality_remote.py --prepare preparation.json
python -B scripts/native_dataset_quality_remote.py --execute /report/case/ACTION.json
```

The remote command unsets `ONNX_SPLITPOINT_ARTIFACT_POLICY`: its Generic
`cache_verify_only` value marks Native consumer attestations as diagnostic.
Native no-build enforcement is independent, using existing sealed artifacts,
direct NativeTRT opening and `ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0`.

An H8 timed result may carry its model only inside the sealed Native command.
The config loader derives that missing alias only after the existing complete
Native command verifier succeeds; present aliases must agree. The launcher
also reuses the exact process-local Hailo site directories from that verified
command in `SPLITPOINT_EXTRA_SITES` for the child process. A supplied conflicting
site list is rejected; global Python environments are not modified.

Focused CPU tests exercise actual existing FIFO/completion functions with
injected hardware, distinct image geometries, export/normal-loader integration,
negative evidence/source/image cases, isolated cold-stage imports, immutable CPU
reference reuse and controller capacity/lease cleanup. These tests are software
evidence; actual hardware capture, statistics and adoption have separate receipts.
