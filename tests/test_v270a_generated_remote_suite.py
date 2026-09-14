from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _assert_generated_runner_is_self_consistent,
)
from onnx_splitpoint_tool.benchmark.suite_refresh import (
    assert_generated_runner_is_self_consistent,
)
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.remote.bundle import (
    build_suite_bundle,
    remote_minimal_bundle_patterns,
)
from onnx_splitpoint_tool.split_export_runners import (
    write_runner_skeleton_onnxruntime,
)


@pytest.mark.parametrize(
    "self_check",
    [
        _assert_generated_runner_is_self_consistent,
        assert_generated_runner_is_self_consistent,
    ],
)
def test_generated_runner_self_check_rejects_late_suite_bootstrap(
    tmp_path: Path,
    self_check,
) -> None:
    runner = tmp_path / "run_split_onnxruntime.py"
    runner.write_text(
        textwrap.dedent(
            """
            from splitpoint_runners.native_split_quality_runtime import helper

            def _maybe_add_suite_runtime_to_syspath():
                return None

            _maybe_add_suite_runtime_to_syspath()
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="before the suite root is added to sys.path",
    ):
        self_check(runner)


@pytest.mark.runtime
def test_generated_remote_suite_is_self_contained_for_quality_and_yolo_postprocess(
    tmp_path: Path,
) -> None:
    """Exercise the actual generated-and-bundled suite without the main package.

    This deliberately uses a child interpreter.  In-process import tests can
    accidentally succeed through the already imported source checkout and did
    not catch the top-level ``splitpoint_runners`` packaging regression.
    """

    pytest.importorskip(
        "onnxruntime",
        reason="remote runtime probe requires the optional runtime extra",
    )

    source_suite = tmp_path / "generated_suite"
    case_dir = source_suite / "b038"
    case_dir.mkdir(parents=True)
    (source_suite / "benchmark_set.json").write_text(
        json.dumps({
            "model_name": "yolov7_paper",
            "benchmark_task": "detection",
            "cases": [{"case_id": "b038"}],
        }),
        encoding="utf-8",
    )

    write_benchmark_suite_script(source_suite)
    write_runner_skeleton_onnxruntime(str(case_dir), target="cpu")

    includes, excludes = remote_minimal_bundle_patterns()
    archive = tmp_path / "generated_remote_suite.tar.gz"
    build_suite_bundle(
        source_suite,
        archive,
        includes=includes,
        excludes=excludes,
    )

    extracted_suite = tmp_path / "remote_host" / "suite"
    extracted_suite.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = set(handle.getnames())
        for member in handle.getmembers():
            target = (extracted_suite / member.name).resolve()
            assert target == extracted_suite or extracted_suite in target.parents
            assert not member.issym() and not member.islnk()
        if sys.version_info >= (3, 12):
            handle.extractall(extracted_suite, filter="data")
        else:  # pragma: no cover - compatibility with older supported Python
            handle.extractall(extracted_suite)

    required_members = {
        "benchmark_suite.py",
        "b038/run_split_onnxruntime.py",
        "splitpoint_runners/native_command_contract.py",
        "splitpoint_runners/native_split_quality.py",
        "splitpoint_runners/native_split_quality_runtime.py",
        "splitpoint_runners/native_detection_postprocess.py",
    }
    assert required_members <= members

    # Execute the actual case entry point exactly as the remote suite does:
    # from outside the suite, with no suite PYTHONPATH and with the management
    # package explicitly forbidden.  The runner itself must discover the
    # suite-owned ``splitpoint_runners`` package before its first import.
    blocker_dir = tmp_path / "entrypoint_blocker"
    blocker_dir.mkdir()
    blocker_marker = blocker_dir / "sitecustomize_active.txt"
    (blocker_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""
            import importlib.abc
            from pathlib import Path
            import sys

            Path({str(blocker_marker)!r}).write_text("active\\n", encoding="utf-8")

            class DenyInstalledTool(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if (
                        fullname == "onnx_splitpoint_tool"
                        or fullname.startswith("onnx_splitpoint_tool.")
                    ):
                        raise ModuleNotFoundError(
                            "installed main package is forbidden in remote entrypoint probe: "
                            + fullname
                        )
                    return None

            sys.meta_path.insert(0, DenyInstalledTool())
            """
        ),
        encoding="utf-8",
    )
    foreign_cwd = tmp_path / "remote_workdir"
    foreign_cwd.mkdir()
    entrypoint_environment = os.environ.copy()
    entrypoint_environment.pop("PYTHONHOME", None)
    entrypoint_environment["PYTHONNOUSERSITE"] = "1"
    entrypoint_environment["PYTHONPATH"] = str(blocker_dir)
    entrypoint = subprocess.run(
        [
            sys.executable,
            str(extracted_suite / "b038" / "run_split_onnxruntime.py"),
            "--help",
        ],
        cwd=foreign_cwd,
        env=entrypoint_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert blocker_marker.read_text(encoding="utf-8") == "active\n"
    assert entrypoint.returncode == 0, (
        f"remote case entrypoint failed with rc={entrypoint.returncode}\n"
        f"stdout:\n{entrypoint.stdout}\n"
        f"stderr:\n{entrypoint.stderr}"
    )
    assert "usage:" in entrypoint.stdout.lower()

    # A real venv invocation is important here: its python executable is often
    # a symlink to the system binary, but only the un-resolved venv path adds
    # the Hailo environment's site-packages.
    hailo_venv = tmp_path / "fake_hailo_venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(hailo_venv)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    hailo_python = hailo_venv / "bin" / "python3"
    site_probe = subprocess.run(
        [str(hailo_python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert site_probe.returncode == 0, site_probe.stderr
    hailo_site = Path(site_probe.stdout.strip())
    (hailo_site / "hailo_platform.py").write_text(
        """
class _Quant:
    qp_scale = 0.25
    qp_zp = 7.0

class _Output:
    name = "fake_hef_output"
    shape = (1, 4, 5)
    quant_info = _Quant()

class HEF:
    def __init__(self, path):
        self.path = path
    def get_output_vstream_infos(self):
        return [_Output()]
""".strip() + "\n",
        encoding="utf-8",
    )

    probe = textwrap.dedent(
        r"""
        import importlib
        import importlib.abc
        import json
        from pathlib import Path
        import sys

        import numpy as np


        class DenyInstalledTool(importlib.abc.MetaPathFinder):
            def __init__(self):
                self.attempts = []

            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname == "onnx_splitpoint_tool"
                    or fullname.startswith("onnx_splitpoint_tool.")
                ):
                    self.attempts.append(fullname)
                    raise ModuleNotFoundError(
                        "installed main package is forbidden in remote-suite probe: "
                        + fullname
                    )
                return None


        suite = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(suite))
        blocker = DenyInstalledTool()
        sys.meta_path.insert(0, blocker)

        # Prove that the isolation guard is active, then require the vendored
        # imports below to complete without touching that forbidden namespace.
        try:
            importlib.import_module("onnx_splitpoint_tool")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("onnx_splitpoint_tool unexpectedly importable")
        assert blocker.attempts == ["onnx_splitpoint_tool"]
        blocker.attempts.clear()

        quality_runtime = importlib.import_module(
            "splitpoint_runners.native_split_quality_runtime"
        )
        quality_contract = importlib.import_module(
            "splitpoint_runners.native_split_quality"
        )
        policy = quality_contract.known_native_split_policy(
            model_id="yolo26s",
            case_id="b038",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
        )
        assert policy is not None
        assert policy["task"] == "detection"
        assert policy["precision"] == "uint8_dequant_fp16"
        assert callable(quality_runtime.prepare_native_split_quality_binding)
        assert quality_runtime._builder_script() == (
            suite / "splitpoint_runners" / "native_trt_from_benchmarkset.py"
        )
        fake_hef = suite / "fake_boundary.hef"
        fake_hef.write_bytes(b"fake-hef-for-metadata-only")
        hailo_metadata = quality_runtime._hailo_metadata(
            part1=fake_hef,
            part2_input={"name": "part2_input", "shape": [1, 4, 5]},
            policy={
                "boundary_dtype": "uint8",
                "quantization_policy": "required",
            },
        )
        assert hailo_metadata == {
            "name": "part2_input",
            "runtime_name": "fake_hef_output",
            "shape": [1, 4, 5],
            "canonical_part2_shape": [1, 4, 5],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.25,
                "zero_point": 7.0,
            },
        }

        postprocess = importlib.import_module(
            "splitpoint_runners.native_detection_postprocess"
        )
        yolo_harness = importlib.import_module(
            "splitpoint_runners.harness.yolo"
        )
        outputs = {
            "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
            "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
            "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
        }
        frozen = postprocess.build_frozen_postprocess_contract(
            model_id="yolov7_paper",
            outputs=outputs,
            input_hw=[640, 640],
            original_wh=[1280, 720],
            model_sha256=yolo_harness.YOLOV7_PAPER_ONNX_SHA256,
        )
        processor = postprocess.FrozenDetectionPostprocessor(frozen)
        processed = processor.process(outputs, original_wh=[1280, 720])
        assert processed["task"] == "detection"
        assert processed["contract_family"] == "decoded_nms"
        assert processed["decoder_format"] == "multiscale_head"
        assert processed["detection_count"] == 0
        assert processor.completed_count == 1

        yolo26_outputs = {}
        conv = 61
        for size in (80, 40, 20):
            yolo26_outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
                (size, size, 4), dtype=np.float32
            )
            yolo26_outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
                (size, size, 80), -20.0, dtype=np.float32
            )
            conv += 16
        yolo26_frozen = postprocess.build_frozen_postprocess_contract(
            model_id="yolo26s",
            outputs=yolo26_outputs,
            input_hw=[640, 640],
            original_wh=[1280, 720],
        )
        assert yolo26_frozen["model_family"] == "yolo26"
        assert yolo26_frozen["decoder_format"] == "ultralytics_regcls"

        loaded = {
            "quality_runtime": quality_runtime,
            "quality_contract": quality_contract,
            "postprocess": postprocess,
            "yolo_harness": sys.modules["splitpoint_runners.harness.yolo"],
            "harness_base": sys.modules["splitpoint_runners.harness.base"],
        }
        loaded_paths = {}
        for name, module in loaded.items():
            path = Path(module.__file__).resolve()
            assert path == suite or suite in path.parents, (name, path, suite)
            loaded_paths[name] = str(path.relative_to(suite))

        assert blocker.attempts == [], blocker.attempts
        print(json.dumps({
            "status": "ok",
            "policy_sha256": policy["policy_sha256"],
            "postprocess_contract_sha256": frozen["contract_sha256"],
            "yolo26_postprocess_contract_sha256": yolo26_frozen["contract_sha256"],
            "hailo_runtime_name": hailo_metadata["runtime_name"],
            "loaded_paths": loaded_paths,
            "forbidden_import_attempts": blocker.attempts,
        }, sort_keys=True))
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HAILO_PY"] = str(hailo_python)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(extracted_suite)],
        cwd=extracted_suite,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"remote-suite probe failed with rc={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert output_lines, "remote-suite probe produced no result"
    result = json.loads(output_lines[-1])
    assert result["status"] == "ok"
    assert result["forbidden_import_attempts"] == []
    assert result["hailo_runtime_name"] == "fake_hef_output"
    assert len(result["yolo26_postprocess_contract_sha256"]) == 64
    assert result["loaded_paths"] == {
        "harness_base": "splitpoint_runners/harness/base.py",
        "postprocess": "splitpoint_runners/native_detection_postprocess.py",
        "quality_contract": "splitpoint_runners/native_split_quality.py",
        "quality_runtime": "splitpoint_runners/native_split_quality_runtime.py",
        "yolo_harness": "splitpoint_runners/harness/yolo.py",
    }
