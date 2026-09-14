# Q3 Generic Full original evidence

The JSON projection contains literal selected fields of every original MobileNet normalized row and the original required identities. CSV fixtures retain scalar columns from the original Hailo Full CSVs; their nested multi-megabyte quality/sample cells are omitted. They are evidence projections, not new measurements. The DeepX blocked row and three dispatch/copy-manifest pairs are copied byte for byte. Original files under v33_inputs/Q3 are read only. Tests label modified contradictory contexts and temporary path relocation as synthetic.

Observed defect: Hailo Full source setup_id and run_cfg.quality_evidence_setup_id are empty, while the actual dispatch args carry the correct setup. _run_case forwarded dispatch identity only for composed Generic central-quality rows. classification_logits was already reported by the real runner. An absent DeepX runtime endpoint is not a measured endpoint; the existing normalizer's decoded_or_native default does not change that.

The yolo11l subdirectory retains matching scalar-column projections from both original JSON and CSV files. The actual Full measurement endpoint is completed_detection and the observed latency is 87.1666431427002 ms; it is a mirror regression, not an additional hardware result.
