from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.gui.hailo_diagnostics import (
    collect_hailo_diagnostics,
    format_hailo_diagnostics_short_lines,
    format_hailo_diagnostics_text,
    load_hailo_result_json,
)


def test_collect_hailo_diagnostics_promotes_extra_fields() -> None:
    payload = {
        "ok": True,
        "elapsed_s": 530.25,
        "hw_arch": "hailo8",
        "net_name": "yolov7_part1_b80",
        "backend": "venv",
        "details": {
            "process_summary": {
                "context_count": 1,
                "partition_iterations": 4,
                "partition_time_s": 40.0,
                "compilation_time_s": 4.0,
                "snr_db": {"yolov7_part1_b80/output_layer1": 16.28},
                "detected": {"normalization_warning": True},
            },
            "extra_fields": {
                "result_json_path": "/tmp/hailo_hef_build_result.json",
                "cuda_probe": {"summary": "Compute: GPU (auto)"},
            },
        },
    }

    entry = collect_hailo_diagnostics(payload, label="benchmark b80 part1 @ hailo8")

    assert entry["label"] == "benchmark b80 part1 @ hailo8"
    assert entry["status"] == "OK"
    assert entry["paths"]["result_json"] == "/tmp/hailo_hef_build_result.json"
    assert entry["cuda_probe"]["summary"] == "Compute: GPU (auto)"
    assert entry["process_summary"]["context_count"] == 1


def test_format_hailo_diagnostics_short_lines_include_core_summary() -> None:
    entry = collect_hailo_diagnostics(
        {
            "ok": False,
            "elapsed_s": 3145.13,
            "hw_arch": "hailo8",
            "backend": "venv",
            "error": "Compilation timed out after 10800s\nmore details...",
            "details": {
                "process_summary": {
                    "context_count": 4,
                    "partition_iterations": 327,
                    "partition_time_s": 1568.215,
                    "allocation_time_s": 1975.0,
                    "compilation_time_s": 64.0,
                    "snr_db": {
                        "out1": 28.08,
                        "out2": 26.70,
                    },
                    "detected": {
                        "single_context_failed": True,
                        "multi_context_used": True,
                    },
                },
                "extra_fields": {"result_json_path": "/tmp/result.json"},
            },
        },
        label="suite full @ hailo8",
    )

    text = "\n".join(format_hailo_diagnostics_short_lines(entry))
    assert "suite full @ hailo8" in text
    assert "contexts=4" in text
    assert "327 iters" in text
    assert "single_context_failed" in text
    assert "result_json: /tmp/result.json" in text
    assert "Compilation timed out" in text



def test_format_hailo_diagnostics_text_renders_sections() -> None:
    entry = collect_hailo_diagnostics(
        {
            "ok": True,
            "elapsed_s": 600.6,
            "hw_arch": "hailo8",
            "net_name": "yolov7_part2_b80",
            "backend": "venv",
            "calib_info": {"source": "/data/coco", "used_count": 16, "requested_count": 16, "batch_size": 8},
            "fixup_report": {"kernel_shape_patched": 25, "conv_defaults_added": 0},
            "details": {
                "process_summary": {
                    "context_count": 3,
                    "partition_iterations": 237,
                    "partition_time_s": 865.6,
                    "allocation_time_s": 40.0,
                    "compilation_time_s": 4.0,
                    "stage_durations_s": {"translation": 2.6, "bias_correction": 177.3},
                    "algo_times_s": {"Bias Correction": 177.26, "Layer Noise Analysis": 62.82},
                    "snr_db": {
                        "yolov7_part2_b80/output_layer1": 31.49,
                        "yolov7_part2_b80/output_layer2": 30.09,
                    },
                    "detected": {"multi_context_used": True, "normalization_warning": True},
                },
                "system_snapshot": {
                    "platform": "linux",
                    "python": "3.10",
                    "commands": {
                        "nvidia_smi": {"returncode": 0, "stdout": "GPU 0: RTX 3080 Ti"},
                    },
                },
            },
        },
        label="benchmark b80 part2 @ hailo8",
    )

    text = format_hailo_diagnostics_text(entry)
    assert "Label: benchmark b80 part2 @ hailo8" in text
    assert "Calibration:" in text
    assert "Fixup report:" in text
    assert "Process summary:" in text
    assert "partition search:" in text
    assert "optimization algorithms:" in text
    assert "SNR:" in text
    assert "System snapshot:" in text
    assert "GPU 0: RTX 3080 Ti" in text



def test_load_hailo_result_json_reads_payload(tmp_path: Path) -> None:
    path = tmp_path / "hailo_hef_build_result.json"
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": 12.5,
                "hw_arch": "hailo8",
                "net_name": "demo",
                "backend": "venv",
                "details": {"process_summary": {"context_count": 2}},
            }
        ),
        encoding="utf-8",
    )

    entry = load_hailo_result_json(path)

    assert entry["label"] == tmp_path.name
    assert entry["paths"]["result_json"] == str(path)
    assert entry["process_summary"]["context_count"] == 2
