# Hailo10 physical output fixtures

Six exact UINT8 output buffers from the uploaded v2.80.4 night smoke (three images each for YOLO26m b398 and YOLO26s b364). The source ZIP and every original NPZ member are identified in `manifest.json`. Only B buffers are copied; images and model files are excluded.

These inputs test the software mapping and unchanged existing bridge. They do not confer a quality pass: all recorded endpoint scores were zero, and no network or compiler tuning is applied. The regression uses the production bridge builder around an identity continuation to observe the full boundary tensor; the original TensorRT engine is not rerun locally.
