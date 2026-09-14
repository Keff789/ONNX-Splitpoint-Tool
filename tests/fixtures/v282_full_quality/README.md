# Original September 13 evidence

Source: `completsetdev_20260913_123411_debug_pack` from the plan-associated run.

`original_yolo26m.json` contains unmodified JSON objects extracted from:
- both native_producers/{hailo8,hailo10h}/quality_first/vendor_full_quality_request_binding_set.json files;
- both analysis_tables/native_full_baseline_eval.json YOLO26m Full rows;
- reports/native_validation/native_producer_validation_summary.json corresponding rows and accuracy policy;
- quality_management/central_quality_summary.json exact results selected by their already recorded central result SHA256.

`original_status_evidence.json` contains original Native evidence status; original blocking reason records (only the repeated nested native_evidence_status object is omitted); original four YOLOv7 dump-conflict validation rows; and all eleven nonexecuted Native result rows from reports/native_producer_summary.json.

No tensor, decision, metric, identity, negative evidence or threshold was modified. Formatting and containment differ from the original files. Tests generate derived bindings from the original central result's completed endpoint using the production v2.82 binding projection. This is offline evidence transport verification, not a new runtime measurement or quality evaluation. The original four dump conflicts remain invalid.
