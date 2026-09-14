# Native Boundary Interface Contract Validator

`native_boundary_interface_contract_validator.py` compares a native raw boundary dump against the ONNX Part1 activation for the exact same input image. It evaluates global affine dequantization, direct per-channel affine fits, and channel reorder/layout hypotheses.

Example:

```bash
python scripts/native_boundary_interface_contract_validator.py \
  --benchmark-set "$BS" \
  --case b038 \
  --boundary-manifest "$BS/native_pipeline/b038/hailo_to_trt/uint8_dequant_fp16/native_fifo_boundary/native_fifo_boundary_manifest.json" \
  --reference-report /tmp/yolo26s_b038_ort_cpu_validation_report.json \
  --require-image-match
```

The report classifies the contract as one of:

- `global_dequant_candidate`
- `per_channel_dequant_candidate`
- `layout_or_channel_reorder_suspect`
- `possible_reorder_or_layout_transform`
- `not_same_tensor_or_transform`
- `insufficient_evidence`
