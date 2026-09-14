# Hold-out selection record

Complete this record before any corresponding hardware measurements are opened.

The endpoint-family adapter, preprocessing, layout/dequant provenance,
decoder, NMS and their implementation-source hashes must already be locked in
the pipeline contract manifest. Compatibility work may use graph metadata,
public export specifications, synthetic tensors and development models only.
If a hold-out result influences an adapter change, reclassify that exact model
as development and confirm the amended protocol on a fresh untouched model.

## Classification hold-out

- Model ID:
- Model family:
- Exact ONNX/model hash:
- Why it is independent from the development models:
- Previously inspected benchmark results: no / yes (if yes, it is not a strict hold-out)
- Frozen adapter contract ID and implementation-source hash:
- Candidate universe: `all_feasible` / `deterministic_audit`
- Attested by:
- Attested at (ISO 8601):

## Detection hold-out

- Model ID:
- Model family:
- Exact ONNX/model hash:
- Why it is independent from the development models:
- Previously inspected benchmark results: no / yes (if yes, it is not a strict hold-out)
- Frozen endpoint-family adapter ID and implementation-source hash:
- Candidate universe: `all_feasible` / `deterministic_audit`
- Audit size and seed, if applicable:
- Attested by:
- Attested at (ISO 8601):
