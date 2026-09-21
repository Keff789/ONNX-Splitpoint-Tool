from __future__ import annotations

import inspect
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_command_failure_prefix,
    _remote_preflight_failure,
    _remote_run_capacity_requirement,
    _remote_run_capacity_requirement_for_args,
    _remote_trt_builder_abi,
    _remote_trt_cache_retention_command,
    _run_mentions_tensorrt,
    _serialized_remote_storage_by_host,
    _stable_trt_engine_cache_key,
    _stable_suite_cache_key,
    _trt_engine_builder_abi_contract,
    _terminal_remote_storage_stage_payload,
    _terminal_remote_failure_payload,
    _verified_uncached_suite_extract_command,
    RemoteBenchmarkArgs,
)
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport
from onnx_splitpoint_tool.workflow import runner as runner_module
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _terminal_remote_failure_from_execution_artifacts_v27516,
)


class V27516PrimaryRemoteErrorCachePolicyTests(unittest.TestCase):
    @staticmethod
    def _write_managed_owner(
        namespace: Path,
        *,
        last_used: float,
        builder_abi_sha256: str = "",
    ) -> None:
        namespace.mkdir(parents=True, exist_ok=True)
        (namespace / ".splitpoint_trt_cache_owner.json").write_text(
            json.dumps(
                {
                    "schema": "onnx-splitpoint/managed-trt-cache-owner",
                    "schema_version": 1,
                    "owner": "onnx-splitpoint-tool",
                    "cache_key": namespace.name,
                    "created_at_unix": last_used,
                    "last_used_at_unix": last_used,
                    "trt_builder_abi_sha256": builder_abi_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_valid_engine_receipt(namespace: Path, *, payload: bytes = b"engine") -> None:
        engine = namespace / "b001" / "engine.plan"
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(payload)
        receipt = {
            "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
            "schema_version": 1,
            "build_returncode": 0,
            "dry_run": False,
            "engine": str(engine.resolve()),
            "engine_sha256": hashlib.sha256(payload).hexdigest(),
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        (engine.parent / "engine_build_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _run_retention(
        base: Path,
        *,
        current_key: str,
        max_namespaces: int,
        max_bytes: int,
        planned_growth: int = 0,
        legacy_suite_key: str = "",
        stable_engine_key: str = "",
        builder_abi_sha256: str = "",
        current_trtexec_sha256: str = "",
        native_trt_precision: str = "fp16",
        native_trt_workspace_mb: int = 4096,
        canonical_full_onnx_sha256: tuple[str, ...] = (),
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        command = _remote_trt_cache_retention_command(
            remote_base=str(base),
            current_key=current_key,
            max_namespaces=max_namespaces,
            max_bytes=max_bytes,
            planned_current_growth_bytes=planned_growth,
            legacy_suite_key=legacy_suite_key,
            stable_engine_key=stable_engine_key,
            builder_abi_sha256=builder_abi_sha256,
            current_trtexec_sha256=current_trtexec_sha256,
            native_trt_precision=native_trt_precision,
            native_trt_workspace_mb=native_trt_workspace_mb,
            canonical_full_onnx_sha256=canonical_full_onnx_sha256,
        )
        completed = subprocess.run(
            ["bash", "-c", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        marker = "SPLITPOINT_TRT_RETENTION_JSON="
        payload: dict[str, object] = {}
        for line in completed.stdout.splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker):])
        return completed, payload

    def _remote_status(
        self,
        path: Path,
        *,
        primary_error: str,
        occurred_at: str,
        rc: int = 70,
        terminal: bool = True,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "hardware_target_id": path.stem,
                    "remote_output": {
                        "remote_rc": rc,
                        "terminal_remote_failure": terminal,
                        "local_run_dir": str(path.parent / "remote-local"),
                        "primary_failure": {
                            "terminal_remote_failure": terminal,
                            "failure_kind": "terminal_remote_execution_failure",
                            "remote_rc": rc,
                            "primary_error": primary_error,
                            "primary_error_context": [primary_error],
                            "primary_error_at": occurred_at,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def test_rc70_preserves_original_enospc_as_primary_error(self) -> None:
        payload = _terminal_remote_failure_payload(
            remote_rc=70,
            recent_remote_lines=[
                "[resnet50] starting benchmark",
                "tee: logs/stderr.txt: No space left on device",
                "OSError: [Errno 28] No space left on device",
                "[remote] cleanup unproven; lease session poisoned",
            ],
            fallback_error="Remote benchmark failed (rc=70)",
            occurred_at="2026-08-06T19:04:40Z",
        )
        self.assertTrue(payload["terminal_remote_failure"])
        self.assertEqual(
            payload["primary_error"],
            "tee: logs/stderr.txt: No space left on device",
        )
        self.assertNotIn(
            "lease session poisoned",
            "\n".join(payload["primary_error_context"]),
        )
        self.assertEqual(
            _terminal_remote_failure_payload(
                remote_rc=130,
                recent_remote_lines=["cancelled"],
            ),
            {},
        )
        preflight_error = _remote_preflight_failure(
            2,
            "tee: logs/stderr.txt: No space left on device",
        )
        self.assertIn("terminal_remote_storage_failure", str(preflight_error))
        self.assertIn("No space left on device", str(preflight_error))

    def test_execution_artifacts_choose_first_rc70_and_ignore_rc130(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            later = self._remote_status(
                root / "remote_benchmark_status_hailo8.json",
                primary_error="later terminal error",
                occurred_at="2026-08-06T19:04:42Z",
            )
            first = self._remote_status(
                root / "remote_benchmark_status_hailo10h.json",
                primary_error="No space left on device",
                occurred_at="2026-08-06T19:04:40Z",
            )
            cancelled = self._remote_status(
                root / "remote_benchmark_status_deepx.json",
                primary_error="cancelled sibling",
                occurred_at="2026-08-06T19:04:41Z",
                rc=130,
                terminal=False,
            )
            result = _terminal_remote_failure_from_execution_artifacts_v27516(
                {
                    "remote_benchmark_status_hailo8_json": later,
                    "remote_benchmark_status_hailo10h_json": first,
                    "remote_benchmark_status_deepx_json": cancelled,
                }
            )
            self.assertEqual(result["primary_error"], "No space left on device")
            self.assertEqual(result["remote_rc"], 70)

    def test_poisoned_remote_stage_stops_before_every_follow_on_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_id = "resnet50"
            benchmark_set = root / "models" / model_id / "benchmark_set"
            benchmark_set.mkdir(parents=True)
            (benchmark_set / "benchmark_plan.json").write_text(
                '{"runs": []}', encoding="utf-8",
            )
            (benchmark_set / "benchmark_set.json").write_text(
                '{"model_name": "resnet50"}', encoding="utf-8",
            )
            status_path = self._remote_status(
                root / "models" / model_id / "benchmark_results"
                / "remote_benchmark_status_hailo10h.json",
                primary_error="tee: logs/stderr.txt: No space left on device",
                occurred_at="2026-08-06T19:04:40Z",
            )
            execution = SimpleNamespace(
                artifacts={
                    "remote_benchmark_status_hailo10h_json": status_path,
                },
                metrics={"remote_requested": True, "remote_dispatched": True},
                status="partial",
                message="partial",
            )

            workflow = EvaluationWorkflowRunner(SimpleNamespace(), log=mock.Mock())
            workflow.run_dir = root
            workflow.artifact_index = {}
            workflow.artifact_index_path = root / "artifact_index.json"
            workflow.profile_payload = {}
            workflow.options = SimpleNamespace()
            workflow.session_id = "test-session"
            workflow._cancel_event = threading.Event()
            workflow._remote_process_registry = SimpleNamespace(cancelled=True)
            workflow._process_registry = None
            workflow._stop_requested = False
            workflow.log = mock.Mock()
            workflow._schedule_management_cpu_reference = mock.Mock()

            with (
                mock.patch.object(
                    runner_module,
                    "benchmark_set_postcondition_v60v",
                    return_value={
                        "valid": True,
                        "selected_suite_dir": str(benchmark_set),
                    },
                ),
                mock.patch.object(
                    runner_module,
                    "execute_benchmark_suite_if_requested",
                    return_value=execution,
                ),
                mock.patch.object(
                    runner_module,
                    "materialize_backend_artifact_decisions",
                    side_effect=AssertionError("post-build stage must not run"),
                ) as post_build,
            ):
                artifacts, metrics, message, stage_status = (
                    workflow._stage_run_benchmarks(model_id, {"id": model_id})
                )

            self.assertEqual(stage_status, "failed")
            self.assertTrue(workflow._stop_requested)
            self.assertEqual(
                message,
                "tee: logs/stderr.txt: No space left on device",
            )
            self.assertEqual(metrics["failure_kind"], "terminal_remote_execution_failure")
            self.assertTrue(metrics["follow_on_failures_suppressed"])
            self.assertIn("primary_failure_json", artifacts)
            post_build.assert_not_called()
            self.assertTrue(workflow._shutdown_management_services_bounded(timeout_s=2)["finished"])

    def test_suite_transport_and_managed_tensorrt_policy_are_bounded(self) -> None:
        command = _verified_uncached_suite_extract_command(
            remote_bundle="/home/nx/splitpoint_runs/current/suite bundle.tar.gz",
            remote_suite_dir="/home/nx/splitpoint_runs/current/suite",
            bundle_hash="a" * 64,
        )
        self.assertLess(command.index("sha256sum"), command.index("tar -xzf"))
        self.assertLess(command.index("tar -xzf"), command.rindex("rm -f"))
        self.assertNotIn("_onnx_splitpoint_cache", command)

        source = inspect.getsource(remote_run.run_remote_benchmark)
        self.assertIn(
            'ONNX_SPLITPOINT_REMOTE_BUNDLE_CACHE", "0"',
            source,
        )
        self.assertIn("cache_enabled = False", source)
        self.assertIn(
            'tensorrt_managed_v27516/{trt_engine_cache_key}',
            source,
        )
        self.assertIn("suite_cache_key = _stable_suite_cache_key(suite_dir)", source)
        self.assertLess(
            source.index('stage="pre_remote_mutation"'),
            source.index('stage="remote_mkdir"'),
        )
        self.assertLess(
            source.index('stage="before_direct_suite_upload"'),
            source.index('Uploading suite (direct scp -r)'),
        )
        self.assertIn('failure_kind="terminal_remote_storage_failure"', source)
        self.assertIn("no collection/SCP follows", source)
        self.assertIn("planned_current_growth_bytes", source)
        self.assertIn("if trt_cache_active:", source)
        self.assertIn("if not trt_cache_active:", source)
        self.assertIn(
            "selected plan has no TensorRT workload; managed TRT namespace is not created",
            source,
        )
        self.assertLess(
            source.index('stage="remote_trt_cache_retention"'),
            source.index("Running benchmark suite on remote"),
        )
        self.assertLess(
            source.index("collect_storage_failure"),
            source.index('log("Packaging results on remote")'),
        )
        self.assertLess(
            source.index("pack_storage_failure"),
            source.index('log("Downloading results (scp)")'),
        )
        self.assertLess(
            source.index(
                "if terminal_remote_failure or terminal_remote_storage_failure"
            ),
            source.index('log("Collecting results on remote (best effort)")'),
        )
        self.assertIn(
            "rc, out = transport.run_read_only(cmd, timeout=90 if deep else 30)",
            source,
        )

        lease_scope = mock.Mock()
        transport = SSHTransport(
            HostConfig(id="test", label="test", host="example.invalid"),
            remote_lease_scope=lease_scope,
        )
        with mock.patch.object(
            transport,
            "_run_capture",
            return_value=(0, "/remote/base\n"),
        ) as capture:
            self.assertEqual(
                transport.resolve_path_read_only("~/splitpoint_runs"),
                "/remote/base",
            )
            lease_scope.operation.assert_not_called()
            self.assertIsNone(capture.call_args.kwargs.get("remote_operation"))
            capture.reset_mock(return_value=True)
            capture.return_value = (0, "diagnostic\n")
            self.assertEqual(
                transport.run_read_only("df -hP /remote/base"),
                (0, "diagnostic\n"),
            )
            lease_scope.operation.assert_not_called()
            self.assertIsNone(capture.call_args.kwargs.get("remote_operation"))

    def test_semantic_tensorrt_cache_key_ignores_run_metadata_but_binds_graph(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            roots = []
            for label, stamp in (("a", "2026-08-01"), ("b", "2026-08-02")):
                suite = Path(td) / label / "resnet50" / "benchmark_set"
                (suite / "models").mkdir(parents=True)
                (suite / "models" / "model.onnx").write_bytes(b"same-onnx")
                (suite / "benchmark_plan.json").write_text(
                    json.dumps({
                        "created_at": stamp,
                        "suite_dir": f"/different/EvaluationRuns/{label}/suite",
                        "model_suite": {"primary": [{"id": "resnet50"}]},
                        "runs": [{"id": "ort_tensorrt", "precision": "fp16"}],
                    }),
                    encoding="utf-8",
                )
                (suite / "benchmark_set.json").write_text(
                    json.dumps({
                        "model_name": "resnet50",
                        "created_at": stamp,
                        "source_model": f"/different/EvaluationRuns/{label}/model.onnx",
                        "generation_log": {
                            "created_at": stamp,
                            "run_path": f"/different/EvaluationRuns/{label}/logs/build.log",
                        },
                        "runtime_contract": {
                            "builder_policy": "strict",
                            "workspace_mib": 2048,
                        },
                        "cases": [{
                            "case_id": "b052",
                            "hailo_compile": {
                                "hailo8": {
                                    "part1": {
                                        "artifact_hash": "stable-part1-contract",
                                        "elapsed_s": 0.125 if label == "a" else 8.75,
                                    },
                                },
                            },
                        }],
                        "hailo": {
                            "hefs": {
                                "hailo8": {
                                    "full_build": {
                                        "artifact_hash": "stable-full-contract",
                                        "elapsed_s": 1.5 if label == "a" else 19.0,
                                    },
                                },
                            },
                        },
                    }),
                    encoding="utf-8",
                )
                roots.append(suite)
            key_a = _stable_suite_cache_key(roots[0])
            key_b = _stable_suite_cache_key(roots[1])
            self.assertEqual(key_a, key_b)
            changed_set = json.loads(
                (roots[1] / "benchmark_set.json").read_text(encoding="utf-8")
            )
            changed_set["runtime_contract"]["builder_policy"] = "relaxed"
            (roots[1] / "benchmark_set.json").write_text(
                json.dumps(changed_set),
                encoding="utf-8",
            )
            self.assertNotEqual(key_a, _stable_suite_cache_key(roots[1]))
            changed_set["runtime_contract"]["builder_policy"] = "strict"
            (roots[1] / "benchmark_set.json").write_text(
                json.dumps(changed_set),
                encoding="utf-8",
            )
            graph = roots[1] / "models" / "model.onnx"
            before = graph.stat()
            graph.write_bytes(b"evil-onnx")
            os.utime(
                graph,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            self.assertNotEqual(key_a, _stable_suite_cache_key(roots[1]))

    def test_trt_engine_key_excludes_profile_vendor_diagnostics_and_binds_engine_abi(self) -> None:
        builder_abi = {
            "schema": "onnx-splitpoint/remote-trt-builder-abi",
            "schema_version": 1,
            "trtexec_sha256": "a" * 64,
            "trtexec_size_bytes": 12345,
            "trtexec_version": "[08/13/2026-10:00:00] TensorRT v100300",
            "trtexec_version_rc": 0,
            "gpu_identity": ["GPU, uuid, 8.7, driver"],
            "linked_runtime_libraries": [{
                "name": "libnvinfer.so.10",
                "size_bytes": 45678,
                "sha256": "c" * 64,
            }],
        }
        with tempfile.TemporaryDirectory() as td:
            suites = []
            for label, profile, tool_version, cache_hit in (
                ("a", "profile-cold", "2.75.33", False),
                ("b", "profile-warm", "2.75.35", True),
            ):
                suite = Path(td) / label / "resnet50" / "benchmark_set"
                (suite / "models").mkdir(parents=True)
                (suite / "b052").mkdir()
                (suite / "models" / "model.onnx").write_bytes(b"same-full-onnx")
                (suite / "b052" / "model_part2.onnx").write_bytes(b"same-part2-onnx")
                (suite / "benchmark_plan.json").write_text(
                    json.dumps({
                        "campaign": {"id": f"campaign-{label}"},
                        "evaluation_profile": {"profile_id": profile},
                        "model_suite": {"primary": [{"id": "resnet50"}]},
                        "runs": [{
                            "id": "ort_tensorrt",
                            "provider": "tensorrt",
                            "native_trt_precision": "fp16",
                            "native_trt_workspace_mb": 4096,
                        }],
                    }),
                    encoding="utf-8",
                )
                (suite / "benchmark_set.json").write_text(
                    json.dumps({
                        "model": "models/model.onnx",
                        "model_name": "resnet50",
                        "evaluation_profile": {"profile_id": profile},
                        "tool": {"core": tool_version},
                        "cases": [{
                            "case_id": "b052",
                            "hailo_compile": {
                                "hailo8": {"part1": {
                                    "cache_hit": cache_hit,
                                    "skipped": cache_hit,
                                    "context_mode": "skipped" if cache_hit else "single_context_used",
                                }},
                            },
                            "deepx": {
                                "build_status": "ready_reused" if cache_hit else "ready_built",
                            },
                        }],
                    }),
                    encoding="utf-8",
                )
                suites.append(suite)

            suite_key_a = _stable_suite_cache_key(suites[0])
            suite_key_b = _stable_suite_cache_key(suites[1])
            self.assertNotEqual(suite_key_a, suite_key_b)
            engine_key = _stable_trt_engine_cache_key(
                suites[0], builder_abi=builder_abi,
            )
            self.assertEqual(
                engine_key,
                _stable_trt_engine_cache_key(suites[1], builder_abi=builder_abi),
            )

            # Raw probe output is diagnostic and may carry a wall-clock prefix.
            # Neither that timestamp nor GPU UUID is an engine ABI component.
            volatile_probe = dict(builder_abi)
            volatile_probe["trtexec_version"] = (
                "[08/13/2026-10:01:59] TensorRT v100300"
            )
            volatile_probe["trtexec_version_rc"] = 99
            volatile_probe["gpu_identity"] = ["GPU, another-uuid, 8.7, driver"]
            self.assertEqual(
                engine_key,
                _stable_trt_engine_cache_key(
                    suites[0], builder_abi=volatile_probe,
                ),
            )

            # Compare equivalent ABI descriptors under the same runtime
            # contract. Runtime policy is a separate, intentional key input.
            runtime_variant = RemoteBenchmarkArgs(
                add_args="--trt-runtime native_preferred --native-trt-precision fp16 "
                "--native-trt-workspace-mb 4096"
            )
            structured_probe = dict(builder_abi)
            structured_probe["gpu_targets"] = [{
                "index": "17",
                "name": "GPU",
                "compute_capability": "8.7",
                "driver_version": "driver",
            }]
            self.assertEqual(_trt_engine_builder_abi_contract(builder_abi),
                             _trt_engine_builder_abi_contract(structured_probe))
            self.assertEqual(
                engine_key,
                _stable_trt_engine_cache_key(
                    suites[0], args=runtime_variant, builder_abi=structured_probe,
                ),
            )

            # Candidate churn does not invalidate overlapping content-addressed
            # leaves when the canonical Full model is unchanged.
            (suites[1] / "b052" / "model_part2.onnx").write_bytes(b"changed-split")
            self.assertEqual(
                engine_key,
                _stable_trt_engine_cache_key(suites[1], builder_abi=builder_abi),
            )
            self.assertNotEqual(
                engine_key,
                _stable_trt_engine_cache_key(
                    suites[0],
                    args=RemoteBenchmarkArgs(
                        add_args="--native-trt-precision fp32 --native-trt-workspace-mb 4096"
                    ),
                    builder_abi=builder_abi,
                ),
            )
            changed_builder = dict(builder_abi)
            changed_builder["trtexec_sha256"] = "b" * 64
            self.assertNotEqual(
                engine_key,
                _stable_trt_engine_cache_key(suites[0], builder_abi=changed_builder),
            )
            changed_library = json.loads(json.dumps(builder_abi))
            changed_library["linked_runtime_libraries"][0]["sha256"] = "d" * 64
            self.assertNotEqual(
                engine_key,
                _stable_trt_engine_cache_key(
                    suites[0], builder_abi=changed_library,
                ),
            )
            changed_gpu_target = dict(builder_abi)
            changed_gpu_target["gpu_identity"] = [
                "GPU, uuid, 8.7, another-driver"
            ]
            self.assertNotEqual(
                engine_key,
                _stable_trt_engine_cache_key(
                    suites[0], builder_abi=changed_gpu_target,
                ),
            )
            (suites[1] / "models" / "model.onnx").write_bytes(b"changed-full-onnx")
            self.assertNotEqual(
                engine_key,
                _stable_trt_engine_cache_key(suites[1], builder_abi=builder_abi),
            )

    def test_trt_engine_key_binds_selected_gpu_not_unselected_host_inventory(self) -> None:
        base_probe = {
            "trtexec_sha256": "a" * 64,
            "trtexec_size_bytes": 12345,
            "linked_runtime_libraries": [{
                "name": "libnvinfer.so.10",
                "size_bytes": 45678,
                "sha256": "c" * 64,
            }],
            "gpu_targets": [
                {
                    "index": "0", "name": "NVIDIA RTX A6000",
                    "compute_capability": "8.6", "driver_version": "570.10",
                },
                {
                    "index": "1", "name": "NVIDIA L4",
                    "compute_capability": "8.9", "driver_version": "570.10",
                },
            ],
            "selected_gpu_target": {
                "visible_device_index": "0", "physical_device_index": "0",
                "name": "NVIDIA RTX A6000", "compute_capability": "8.6",
                "driver_version": "570.10", "driver_api_version": "12080",
            },
        }
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "resnet50" / "benchmark_set"
            (suite / "models").mkdir(parents=True)
            (suite / "models" / "model.onnx").write_bytes(b"same-full-onnx")
            (suite / "benchmark_plan.json").write_text(
                json.dumps({
                    "model_suite": {"primary": [{"id": "resnet50"}]},
                    "runs": [{"id": "ort_tensorrt", "provider": "tensorrt"}],
                }),
                encoding="utf-8",
            )
            (suite / "benchmark_set.json").write_text(
                json.dumps({"model": "models/model.onnx", "model_name": "resnet50"}),
                encoding="utf-8",
            )

            key_gpu0 = _stable_trt_engine_cache_key(suite, builder_abi=base_probe)

            # Adding, removing or renumbering a GPU which TensorRT did not
            # select cannot invalidate the selected GPU's engines.
            inventory_churn = json.loads(json.dumps(base_probe))
            inventory_churn["gpu_targets"][1].update({
                "index": "17", "name": "NVIDIA H100",
                "compute_capability": "9.0",
            })
            self.assertEqual(
                key_gpu0,
                _stable_trt_engine_cache_key(suite, builder_abi=inventory_churn),
            )

            # Selecting the other CUDA-visible target must create a different
            # namespace even though the host inventory itself is unchanged.
            selected_gpu1 = json.loads(json.dumps(base_probe))
            selected_gpu1["selected_gpu_target"] = {
                "visible_device_index": "0", "physical_device_index": "1",
                "name": "NVIDIA L4", "compute_capability": "8.9",
                "driver_version": "570.10", "driver_api_version": "12080",
            }
            self.assertNotEqual(
                key_gpu0,
                _stable_trt_engine_cache_key(suite, builder_abi=selected_gpu1),
            )

            # A mere physical index renumber does not change serialized-engine
            # compatibility when the selected target's ABI is identical.
            renumbered_selected = json.loads(json.dumps(base_probe))
            renumbered_selected["selected_gpu_target"]["physical_device_index"] = "12"
            self.assertEqual(
                key_gpu0,
                _stable_trt_engine_cache_key(suite, builder_abi=renumbered_selected),
            )

    def test_remote_trt_abi_uses_cuda_driver_api_when_jetson_has_no_nvidia_smi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "fake_cuda.c"
            source.write_text(
                r'''
#include <string.h>
int cuInit(unsigned int flags) { (void)flags; return 0; }
int cuDeviceGetCount(int *count) { *count = 1; return 0; }
int cuDeviceGet(int *device, int ordinal) {
    if (ordinal != 0) return 1; *device = 0; return 0;
}
int cuDeviceGetName(char *name, int len, int device) {
    (void)device; const char *value = "NVIDIA Jetson Orin NX";
    if (len <= 0) return 1; strncpy(name, value, (unsigned long)len - 1);
    name[len - 1] = '\0'; return 0;
}
int cuDeviceComputeCapability(int *major, int *minor, int device) {
    (void)device; *major = 8; *minor = 7; return 0;
}
int cuDriverGetVersion(int *version) { *version = 12040; return 0; }
int cuDeviceGetAttribute(int *value, int attribute, int device) {
    (void)device;
    if (attribute == 33 || attribute == 34 || attribute == 50) {
        *value = 0; return 0;
    }
    return 1;
}
''',
                encoding="utf-8",
            )
            library = root / "libcuda.so.1"
            compiler = subprocess.run(
                ["cc", "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1",
                 "-o", str(library), str(source)],
                check=False, capture_output=True, text=True,
            )
            if compiler.returncode != 0:
                self.skipTest("C compiler unavailable for fake CUDA driver test")
            trtexec = root / "trtexec"
            trtexec.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'TensorRT v10.3.0'\nexit 0\n",
                encoding="utf-8",
            )
            trtexec.chmod(0o755)

            class LocalReadOnlyTransport:
                command = ""

                def run_read_only(self, command: str, timeout: int = 0):
                    self.command = command
                    env = os.environ.copy()
                    env["PATH"] = str(root) + ":/usr/bin:/bin"
                    env["LD_LIBRARY_PATH"] = str(root)
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    completed = subprocess.run(
                        ["bash", "-c", command], env=env,
                        capture_output=True, text=True, timeout=timeout,
                    )
                    return completed.returncode, completed.stdout + completed.stderr

            transport = LocalReadOnlyTransport()
            payload = _remote_trt_builder_abi(transport)  # type: ignore[arg-type]
            selected = payload["selected_gpu_target"]
            self.assertEqual(payload["gpu_targets"], [])
            self.assertNotEqual(payload["gpu_probe_rc"], 0)
            self.assertTrue(payload["cuda_driver_probe"]["ok"])
            self.assertEqual(selected["name"], "NVIDIA Jetson Orin NX")
            self.assertEqual(selected["compute_capability"], "8.7")
            self.assertEqual(selected["driver_api_version"], "12040")
            self.assertEqual(selected["selection_source"], "cuda_driver_api_visible_device_0")
            contract = _trt_engine_builder_abi_contract(payload)
            self.assertEqual(
                contract["selected_gpu_target"]["name"], "NVIDIA Jetson Orin NX",
            )
            self.assertTrue(contract["linked_runtime_libraries"])
            self.assertNotIn("tegrastats", transport.command)

    def test_remote_trt_abi_fails_closed_without_any_selected_gpu_attestation(self) -> None:
        payload = {
            "trtexec_sha256": "a" * 64,
            "trtexec_size_bytes": 123,
            "gpu_targets": [],
            "selected_gpu_target": {},
            "linked_runtime_libraries": [{
                "name": "libnvinfer.so.10", "size_bytes": 10,
                "sha256": "b" * 64,
            }],
        }

        class EmptyGpuTransport:
            def run_read_only(self, command: str, timeout: int = 0):
                marker = "SPLITPOINT_TRT_BUILDER_ABI="
                return 0, marker + json.dumps(payload, sort_keys=True)

        with self.assertRaisesRegex(RuntimeError, "Could not attest"):
            _remote_trt_builder_abi(EmptyGpuTransport())  # type: ignore[arg-type]

    def test_capacity_requirement_counts_peak_bytes_and_directory_inodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "suite"
            (suite / "nested").mkdir(parents=True)
            (suite / "a.bin").write_bytes(b"a" * 100)
            (suite / "nested" / "b.bin").write_bytes(b"b" * 200)
            requirement = _remote_run_capacity_requirement(suite)
            self.assertEqual(requirement["total_bytes"], 300)
            self.assertEqual(requirement["file_count"], 2)
            self.assertGreaterEqual(requirement["directory_count"], 2)
            self.assertGreaterEqual(
                requirement["required_free_bytes"],
                2 * 300 + 512 * 1024 * 1024,
            )
            self.assertGreaterEqual(requirement["required_free_inodes"], 8192)

    def test_selected_plan_capacity_recognizes_real_trt_run_ids(self) -> None:
        self.assertTrue(_run_mentions_tensorrt("hailo8_to_trt"))
        self.assertTrue(_run_mentions_tensorrt("deepx_m1_to_trt"))
        self.assertTrue(_run_mentions_tensorrt("vendor_trt_fp16"))
        self.assertFalse(_run_mentions_tensorrt("hailo8_to_cpu"))

        with tempfile.TemporaryDirectory() as td:
            suite = Path(td) / "suite"
            case = suite / "b001"
            case.mkdir(parents=True)
            (case / "split_manifest.json").write_text("{}", encoding="utf-8")
            (case / "model.onnx").write_bytes(b"onnx-graph")
            (suite / "benchmark_plan.json").write_text(
                json.dumps({
                    "runs": [
                        {"id": "hailo8_to_trt", "precision": "fp16"},
                        {"id": "deepx_m1_to_trt", "precision": "int8"},
                        {"id": "hailo8_to_cpu", "provider": "cpu"},
                    ],
                }),
                encoding="utf-8",
            )

            trt = _remote_run_capacity_requirement_for_args(
                suite,
                args=RemoteBenchmarkArgs(add_args="--run-id hailo8_to_trt"),
            )
            self.assertEqual(trt["plan_run_count"], 1)
            self.assertEqual(trt["trt_run_count"], 1)
            self.assertEqual(trt["trt_profile_count"], 1)
            self.assertGreater(trt["trt_engine_cache_bytes"], 0)
            self.assertGreater(trt["trt_build_scratch_bytes"], 0)
            self.assertGreater(trt["runtime_output_log_bytes"], 0)
            self.assertGreater(
                trt["cold_required_free_bytes"],
                trt["warm_required_free_bytes"],
            )

            cpu = _remote_run_capacity_requirement_for_args(
                suite,
                args=RemoteBenchmarkArgs(add_args="--run-id=hailo8_to_cpu"),
            )
            self.assertEqual(cpu["trt_run_count"], 0)
            self.assertEqual(cpu["trt_engine_cache_bytes"], 0)
            self.assertEqual(
                cpu["cold_required_free_bytes"],
                cpu["warm_required_free_bytes"],
            )

            forced = _remote_run_capacity_requirement_for_args(
                suite,
                args=RemoteBenchmarkArgs(
                    provider="tensorrt",
                    add_args="--run-id=hailo8_to_cpu",
                ),
            )
            self.assertEqual(forced["trt_run_count"], 1)
            self.assertGreater(forced["trt_profile_count"], 0)
            self.assertGreater(
                forced["cold_required_free_bytes"],
                forced["warm_required_free_bytes"],
            )

    def test_remote_runs_are_serialized_per_host(self) -> None:
        active = 0
        maximum = 0
        state_lock = threading.Lock()
        start = threading.Barrier(3)

        @_serialized_remote_storage_by_host
        def protected(*, host: HostConfig, cancel_event: threading.Event) -> None:
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1

        host = HostConfig(
            id="same-host",
            label="same-host",
            host="192.0.2.10",
            user="nx",
            remote_base_dir="/home/nx/splitpoint_runs",
        )

        def invoke() -> None:
            start.wait(timeout=2)
            protected(host=host, cancel_event=threading.Event())

        workers = [threading.Thread(target=invoke) for _ in range(2)]
        for worker in workers:
            worker.start()
        start.wait(timeout=2)
        for worker in workers:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(maximum, 1)

    def test_storage_tokens_are_terminal_for_collect_and_pack_at_every_rc(self) -> None:
        for rc in (0, 1, 2, 70):
            with self.subTest(rc=rc):
                payload = _terminal_remote_storage_stage_payload(
                    stage="remote_result_pack",
                    rc=rc,
                    output="tar: results.tar.gz: No space left on device",
                )
                self.assertTrue(payload["terminal_remote_failure"])
                self.assertEqual(
                    payload["failure_kind"],
                    "terminal_remote_storage_failure",
                )
                self.assertEqual(payload["remote_rc"], rc)
        self.assertEqual(
            _terminal_remote_storage_stage_payload(
                stage="remote_result_collect",
                rc=1,
                output="ordinary copy failure",
            ),
            {},
        )

    def test_retention_rc75_is_terminal_storage_admission_without_enospc_token(self) -> None:
        output = (
            'SPLITPOINT_TRT_RETENTION_JSON={"admission_ok":false,'
            '"reason":"managed_trt_cache_limits_blocked_by_active_or_unowned_namespaces"}'
        )
        self.assertEqual(
            _remote_command_failure_prefix(
                stage="remote_trt_cache_retention",
                output=output,
            ),
            "remote_storage_preflight_failed: ",
        )
        self.assertEqual(
            _remote_command_failure_prefix(
                stage="ordinary_remote_command",
                output="ordinary failure",
            ),
            "",
        )

    def test_retention_prunes_only_owned_receipted_oldest_and_inventories_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            legacy = base / "_onnx_splitpoint_cache" / "tensorrt" / "legacy-key"
            legacy.mkdir(parents=True)
            (legacy / "sentinel").write_text("keep", encoding="utf-8")
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            old = managed / "oldest"
            newer = managed / "newer"
            self._write_managed_owner(old, last_used=1.0)
            self._write_valid_engine_receipt(old, payload=b"old")
            self._write_managed_owner(newer, last_used=2.0)
            self._write_valid_engine_receipt(newer, payload=b"new")

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=2,
                max_bytes=1024**3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(old.exists())
            self.assertTrue(newer.exists())
            self.assertTrue((managed / "current").exists())
            self.assertEqual(
                [row["name"] for row in payload["removed"]],
                ["oldest"],
            )
            self.assertFalse(payload["legacy_inventory"]["deleted"])
            self.assertEqual((legacy / "sentinel").read_text(encoding="utf-8"), "keep")

    def test_retention_migrates_only_receipt_and_gpu_verified_legacy_engine_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            legacy_key = "resnet50-1111111111111111"
            stable_key = "resnet50-2222222222222222"
            legacy = managed / legacy_key
            self._write_managed_owner(legacy, last_used=1.0, builder_abi_sha256="b" * 64)
            source_payload = b"source-onnx"
            source_sha256 = hashlib.sha256(source_payload).hexdigest()
            leaf = (
                legacy / "b052" / "native" / source_sha256[:2]
                / source_sha256 / "full" / "fp16"
            )
            leaf.mkdir(parents=True)
            source = leaf / "source.onnx"
            engine = leaf / "full_fp16.engine"
            source.write_bytes(source_payload)
            engine.write_bytes(b"serialized-engine")
            trtexec = Path(td) / "trtexec"
            trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            trtexec.chmod(0o755)
            receipt = {
                "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
                "schema_version": 1,
                "build_returncode": 0,
                "dry_run": False,
                "command": [
                    str(trtexec.resolve()),
                    f"--onnx={source.resolve()}",
                    f"--saveEngine={engine.resolve()}",
                    "--fp16",
                    "--memPoolSize=workspace:4096",
                ],
                "source_onnx": str(source.resolve()),
                "source_onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "engine": str(engine.resolve()),
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "trtexec": str(trtexec.resolve()),
                "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
            }
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            (leaf / "engine_build_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            quality_leaf = (
                legacy / "native_split_quality" / "orin_nx_hailo8_01"
                / "resnet50" / "b052" / "hailo8_to_trt" / ("d" * 64)
                / "engine_cache" / "b052" / "part2" / "fp16"
            )
            quality_leaf.mkdir(parents=True)
            quality_source = quality_leaf / "source_part2.onnx"
            quality_engine = quality_leaf / "part2_fp16.engine"
            quality_source.write_bytes(b"quality-split-source")
            quality_engine.write_bytes(b"quality-split-engine")
            quality_receipt = {
                "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
                "schema_version": 1,
                "build_returncode": 0,
                "dry_run": False,
                "command": [
                    str(trtexec.resolve()),
                    f"--onnx={quality_source.resolve()}",
                    f"--saveEngine={quality_engine.resolve()}",
                    "--fp16",
                    "--memPoolSize=workspace:4096",
                ],
                "source_onnx": str(quality_source.resolve()),
                "source_onnx_sha256": hashlib.sha256(
                    quality_source.read_bytes()
                ).hexdigest(),
                "engine": str(quality_engine.resolve()),
                "engine_sha256": hashlib.sha256(
                    quality_engine.read_bytes()
                ).hexdigest(),
                "trtexec": str(trtexec.resolve()),
                "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
            }
            quality_receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    quality_receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            (quality_leaf / "engine_build_receipt.json").write_text(
                json.dumps(quality_receipt, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            # A special bridge without its binding proof must remain rejected,
            # independently of the valid direct Part2 above.
            bridge_leaf = quality_leaf.parent / "float32_layout_fp16"
            bridge_leaf.mkdir()
            bridge_source = bridge_leaf / "source_part2_float32_layout_bridge.onnx"
            bridge_engine = bridge_leaf / "part2_float32_layout_fp16.engine"
            bridge_source.write_bytes(b"unbound-special-bridge")
            bridge_engine.write_bytes(b"bridge-engine")
            bridge_receipt = dict(quality_receipt)
            bridge_receipt.update(source_onnx=str(bridge_source.resolve()),
                                  engine=str(bridge_engine.resolve()),
                                  source_onnx_sha256=hashlib.sha256(bridge_source.read_bytes()).hexdigest(),
                                  engine_sha256=hashlib.sha256(bridge_engine.read_bytes()).hexdigest(),
                                  command=[str(trtexec.resolve()), f"--onnx={bridge_source.resolve()}",
                                           f"--saveEngine={bridge_engine.resolve()}", "--fp16", "--memPoolSize=workspace:4096"])
            bridge_receipt.pop("receipt_sha256")
            bridge_receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
                bridge_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            (bridge_leaf / "engine_build_receipt.json").write_text(json.dumps(bridge_receipt))
            (legacy / ".active.lock").touch()
            legacy_before = {
                path.relative_to(legacy): path.read_bytes()
                for path in legacy.rglob("*") if path.is_file()
            }

            completed, payload = self._run_retention(
                base,
                current_key=stable_key,
                max_namespaces=6,
                max_bytes=1024**3,
                # Exercise discovery across version/profile suite-key churn;
                # the actual old root is selected by its canonical Full receipt.
                legacy_suite_key="resnet50-3333333333333333",
                stable_engine_key=stable_key,
                builder_abi_sha256="b" * 64,
                current_trtexec_sha256=hashlib.sha256(trtexec.read_bytes()).hexdigest(),
                native_trt_precision="fp16",
                native_trt_workspace_mb=4096,
                canonical_full_onnx_sha256=(source_sha256,),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            migration = payload["legacy_receipt_migration"]
            self.assertEqual(migration["status"], "verified_receipts_migrated")
            self.assertEqual(migration["migrated_receipts"], 2)
            self.assertEqual(migration["rejected_receipts"], 1)
            self.assertEqual(migration["candidate_failures"][0]["reason"], "special_bridge_binding_unverified")
            self.assertFalse(any((managed / stable_key).rglob("part2_float32_layout_fp16.engine")))
            self.assertEqual(migration["source_keys"], [legacy_key])
            self.assertEqual(
                legacy_before,
                {
                    path.relative_to(legacy): path.read_bytes()
                    for path in legacy.rglob("*") if path.is_file()
                },
            )
            migrated_leaf = (
                managed / stable_key / "full" / source_sha256 / "fp16"
            )
            self.assertEqual((migrated_leaf / engine.name).read_bytes(), engine.read_bytes())
            migrated_quality_leaf = (
                managed / stable_key / "splits" / "b052" / "part2"
                / hashlib.sha256(quality_source.read_bytes()).hexdigest()
                / "fp16"
            )
            self.assertEqual(
                (migrated_quality_leaf / quality_engine.name).read_bytes(),
                quality_engine.read_bytes(),
            )
            migrated_receipt = json.loads(
                (migrated_leaf / "engine_build_receipt.json").read_text(encoding="utf-8")
            )
            self.assertTrue(str(migrated_receipt["engine"]).startswith(str(managed / stable_key)))
            self.assertTrue(str(migrated_receipt["source_onnx"]).startswith(str(managed / stable_key)))
            unsigned = dict(migrated_receipt)
            declared = unsigned.pop("receipt_sha256")
            self.assertEqual(
                declared,
                hashlib.sha256(
                    json.dumps(
                        unsigned,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            owner = json.loads(
                (managed / stable_key / ".splitpoint_trt_cache_owner.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(owner["trt_engine_cache_key"], stable_key)
            self.assertEqual(owner["trt_builder_abi_sha256"], "b" * 64)

    def test_retention_can_discover_legacy_root_without_an_exact_suite_key_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            legacy = managed / "resnet50-1111111111111111"
            stable_key = "resnet50-2222222222222222"
            self._write_managed_owner(legacy, last_used=1.0, builder_abi_sha256="b" * 64)
            source_payload = b"canonical-full-onnx"
            source_sha256 = hashlib.sha256(source_payload).hexdigest()
            leaf = (
                legacy / "b052" / "native" / source_sha256[:2]
                / source_sha256 / "full" / "fp16"
            )
            leaf.mkdir(parents=True)
            source = leaf / "source.onnx"
            engine = leaf / "full_fp16.engine"
            source.write_bytes(source_payload)
            engine.write_bytes(b"serialized-engine")
            trtexec = Path(td) / "trtexec"
            trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            trtexec.chmod(0o755)
            receipt = {
                "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
                "schema_version": 1,
                "build_returncode": 0,
                "dry_run": False,
                "command": [
                    str(trtexec.resolve()),
                    f"--onnx={source.resolve()}",
                    f"--saveEngine={engine.resolve()}",
                    "--fp16",
                    "--memPoolSize=workspace:4096",
                ],
                "source_onnx": str(source.resolve()),
                "source_onnx_sha256": source_sha256,
                "engine": str(engine.resolve()),
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "trtexec": str(trtexec.resolve()),
                "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
            }
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(
                    receipt, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            (leaf / "engine_build_receipt.json").write_text(
                json.dumps(receipt, sort_keys=True), encoding="utf-8",
            )

            completed, payload = self._run_retention(
                base,
                current_key=stable_key,
                max_namespaces=6,
                max_bytes=1024**3,
                legacy_suite_key="",
                stable_engine_key=stable_key,
                builder_abi_sha256="b" * 64,
                current_trtexec_sha256=hashlib.sha256(trtexec.read_bytes()).hexdigest(),
                native_trt_precision="fp16",
                native_trt_workspace_mb=4096,
                canonical_full_onnx_sha256=(source_sha256,),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            migration = payload["legacy_receipt_migration"]
            self.assertTrue(migration["requested"])
            self.assertEqual(migration["source_keys"], [legacy.name])
            self.assertEqual(migration["migrated_receipts"], 1)

    def test_retention_rejects_unsealed_abi_workspace_and_gpu_incompatible_legacy_leaves(self) -> None:
        for scenario in (
            "receipt_digest_tampered",
            "builder_binary_mismatch",
            "precision_mismatch",
            "workspace_mismatch",
            "gpu_deserialization_failed",
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "splitpoint_runs"
                managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
                legacy_key = "resnet50-1111111111111111"
                stable_key = "resnet50-2222222222222222"
                legacy = managed / legacy_key
                self._write_managed_owner(legacy, last_used=1.0, builder_abi_sha256="b" * 64)
                source_payload = b"canonical-full-onnx"
                source_sha256 = hashlib.sha256(source_payload).hexdigest()
                leaf_precision = "fp32" if scenario == "precision_mismatch" else "fp16"
                leaf = (
                    legacy / "b052" / "native" / source_sha256[:2]
                    / source_sha256 / "full" / leaf_precision
                )
                leaf.mkdir(parents=True)
                source = leaf / "source.onnx"
                engine = leaf / f"full_{leaf_precision}.engine"
                source.write_bytes(source_payload)
                engine.write_bytes(b"serialized-engine")
                trtexec = Path(td) / "trtexec"
                trtexec.write_text(
                    "#!/bin/sh\nexit 17\n"
                    if scenario == "gpu_deserialization_failed"
                    else "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                trtexec.chmod(0o755)
                workspace = 2048 if scenario == "workspace_mismatch" else 4096
                receipt = {
                    "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
                    "schema_version": 1,
                    "build_returncode": 0,
                    "dry_run": False,
                    "command": [
                        str(trtexec.resolve()),
                        f"--onnx={source.resolve()}",
                        f"--saveEngine={engine.resolve()}",
                        *([] if leaf_precision == "fp32" else ["--fp16"]),
                        f"--memPoolSize=workspace:{workspace}",
                    ],
                    "source_onnx": str(source.resolve()),
                    "source_onnx_sha256": source_sha256,
                    "engine": str(engine.resolve()),
                    "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                    "trtexec": str(trtexec.resolve()),
                    "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
                }
                receipt["receipt_sha256"] = hashlib.sha256(
                    json.dumps(
                        receipt,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                if scenario == "receipt_digest_tampered":
                    receipt["engine_sha256"] = "e" * 64
                receipt_path = leaf / "engine_build_receipt.json"
                receipt_path.write_text(
                    json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8",
                )
                (legacy / ".active.lock").touch()
                old_bytes = {
                    path.relative_to(legacy): path.read_bytes()
                    for path in legacy.rglob("*") if path.is_file()
                }
                current_builder_hash = (
                    "f" * 64
                    if scenario == "builder_binary_mismatch"
                    else hashlib.sha256(trtexec.read_bytes()).hexdigest()
                )

                completed, payload = self._run_retention(
                    base,
                    current_key=stable_key,
                    max_namespaces=6,
                    max_bytes=1024**3,
                    legacy_suite_key=legacy_key,
                    stable_engine_key=stable_key,
                    builder_abi_sha256="b" * 64,
                    current_trtexec_sha256=current_builder_hash,
                    native_trt_precision="fp16",
                    native_trt_workspace_mb=4096,
                    canonical_full_onnx_sha256=(source_sha256,),
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                migration = payload["legacy_receipt_migration"]
                self.assertEqual(migration["migrated_receipts"], 0)
                self.assertEqual(migration["inventoried_receipts"], 1)
                self.assertEqual(migration["rejected_receipts"], 1)
                if scenario == "gpu_deserialization_failed":
                    self.assertEqual(migration["gpu_deserialization_failures"], 1)
                    self.assertEqual(migration["candidate_failures"], [])
                else:
                    self.assertEqual(migration["gpu_deserialization_failures"], 0)
                    expected_reason = {
                        "receipt_digest_tampered": "receipt_digest_invalid",
                        "builder_binary_mismatch": "receipt_builder_abi_mismatch",
                        "precision_mismatch": "receipt_runtime_precision_mismatch",
                        "workspace_mismatch": "receipt_workspace_mismatch",
                    }[scenario]
                    self.assertEqual(migration["candidate_failures"][0]["reason"], expected_reason)
                self.assertFalse(any((managed / stable_key).rglob("*.engine")))
                self.assertEqual(
                    old_bytes,
                    {
                        path.relative_to(legacy): path.read_bytes()
                        for path in legacy.rglob("*") if path.is_file()
                    },
                )

    def test_existing_stable_namespace_uses_shallow_inventory_and_artifact_scan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            legacy_key = "resnet50-1111111111111111"
            stable_key = "resnet50-2222222222222222"
            legacy = managed / legacy_key
            stable = managed / stable_key
            self._write_managed_owner(legacy, last_used=1.0, builder_abi_sha256="b" * 64)
            self._write_valid_engine_receipt(
                legacy,
                payload=b"valid-but-must-not-be-hashed-on-warm-retention",
            )
            self._write_managed_owner(stable, last_used=2.0, builder_abi_sha256="b" * 64)
            sentinel = legacy / "legacy-sentinel.bin"
            sentinel.write_bytes(b"do-not-scan-or-copy")
            unmanaged = base / "_onnx_splitpoint_cache" / "tensorrt" / "old-suite"
            unmanaged.mkdir(parents=True)
            (unmanaged / "large-unmanaged-engine.plan").write_bytes(
                b"unmanaged-tree-must-not-be-deep-inventoried-on-warm-runs"
            )

            completed, payload = self._run_retention(
                base,
                current_key=stable_key,
                max_namespaces=6,
                max_bytes=1024**3,
                legacy_suite_key=legacy_key,
                stable_engine_key=stable_key,
                builder_abi_sha256="b" * 64,
                current_trtexec_sha256="c" * 64,
                canonical_full_onnx_sha256=("d" * 64,),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            migration = payload["legacy_receipt_migration"]
            self.assertTrue(migration["requested"])
            self.assertEqual(
                migration["trigger"],
                "partial_namespace_missing_artifact_scan",
            )
            self.assertTrue(migration["target_preexisting"])
            self.assertEqual(migration["source_keys"], [legacy_key])
            self.assertEqual(migration["inventoried_receipts"], 1)
            self.assertEqual(migration["migrated_receipts"], 0)
            self.assertEqual(migration["rejected_receipts"], 1)
            self.assertEqual(
                payload["retention_fast_path"],
                "existing_stable_namespace_within_limits",
            )
            self.assertFalse(payload["receipt_hash_validation_performed"])
            self.assertEqual(
                payload["legacy_inventory"]["inventory_mode"],
                "shallow_existing_stable_namespace",
            )
            self.assertEqual(
                [row["name"] for row in payload["legacy_inventory"]["entries"]],
                ["old-suite"],
            )
            self.assertEqual(
                [row["name"] for row in payload["retained_unclassified"]],
                [legacy_key],
            )
            self.assertEqual(sentinel.read_bytes(), b"do-not-scan-or-copy")
            self.assertTrue((legacy / ".active.lock").is_file())

    def test_existing_stable_namespace_falls_back_to_verified_pruning_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            old = managed / "old-cache"
            current = managed / "current"
            self._write_managed_owner(old, last_used=1.0)
            self._write_valid_engine_receipt(old, payload=b"old-engine")
            self._write_managed_owner(current, last_used=2.0)

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=1,
                max_bytes=1024**3,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn("retention_fast_path", payload)
            self.assertFalse(old.exists())
            self.assertTrue(current.exists())
            self.assertEqual(
                [row["name"] for row in payload["removed"]],
                ["old-cache"],
            )

    def test_retention_counts_receiptless_owned_namespaces_and_rolls_back_current(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            protected_a = managed / "partial-a"
            protected_b = managed / "partial-b"
            self._write_managed_owner(protected_a, last_used=1.0)
            self._write_managed_owner(protected_b, last_used=2.0)
            for namespace in (protected_a, protected_b):
                filler = namespace / "partial-builder-output.sparse"
                with filler.open("wb") as handle:
                    handle.truncate(600 * 1024**2)

            completed, payload = self._run_retention(
                base,
                current_key="new-current",
                max_namespaces=10,
                max_bytes=1024**3,
            )
            self.assertEqual(completed.returncode, 75, completed.stderr)
            self.assertFalse(payload["admission_ok"])
            self.assertTrue(payload["new_empty_current_rolled_back"])
            self.assertFalse((managed / "new-current").exists())
            self.assertTrue(protected_a.exists())
            self.assertTrue(protected_b.exists())
            self.assertGreater(payload["managed_bytes"], payload["max_bytes"])
            self.assertEqual(
                {row["reason"] for row in payload["protected"]},
                {"no_valid_engine_receipt"},
            )

    def test_irreducible_protected_floor_does_not_delete_eligible_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            for name in ("partial-a", "partial-b"):
                self._write_managed_owner(managed / name, last_used=1.0)
            eligible = managed / "valid-old-cache"
            self._write_managed_owner(eligible, last_used=0.0)
            self._write_valid_engine_receipt(eligible)

            completed, payload = self._run_retention(
                base,
                current_key="new-current",
                max_namespaces=2,
                max_bytes=1024**3,
            )
            self.assertEqual(completed.returncode, 75, completed.stderr)
            self.assertTrue(eligible.exists())
            self.assertEqual(payload["removed"], [])
            self.assertFalse((managed / "new-current").exists())

    def test_regular_foreign_file_bytes_block_admission_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            managed.mkdir(parents=True)
            foreign = managed / ".foreign-builder-output.sparse"
            with foreign.open("wb") as handle:
                handle.truncate(30 * 1024**3)

            completed, payload = self._run_retention(
                base,
                current_key="new-current",
                max_namespaces=6,
                max_bytes=20 * 1024**3,
            )
            self.assertEqual(completed.returncode, 75, completed.stderr)
            self.assertTrue(foreign.exists())
            self.assertFalse((managed / "new-current").exists())
            protected = {row["name"]: row for row in payload["protected"]}
            self.assertEqual(
                protected[foreign.name]["bytes"],
                30 * 1024**3,
            )
            self.assertGreater(payload["managed_bytes"], payload["max_bytes"])

    def test_retention_reserves_planned_current_growth_outside_retained_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            old = managed / "old-cache"
            self._write_managed_owner(old, last_used=1.0)
            self._write_valid_engine_receipt(old)
            filler = old / "builder-cache.sparse"
            with filler.open("wb") as handle:
                handle.truncate(int(19.5 * 1024**3))

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=6,
                max_bytes=20 * 1024**3,
                planned_growth=1024**3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(old.exists())
            self.assertEqual(payload["removed"], [])
            self.assertEqual(payload["planned_current_growth_bytes"], 1024**3)
            self.assertTrue(payload["selected_plan_reserve_applied"])
            self.assertEqual(
                payload["retained_cache_budget_bytes"],
                payload["max_bytes"],
            )
            self.assertLessEqual(
                payload["retained_noncurrent_bytes"],
                payload["retained_cache_budget_bytes"],
            )
            self.assertGreater(
                payload["projected_managed_bytes"],
                payload["max_bytes"],
            )
            self.assertLessEqual(
                payload["projected_managed_bytes"],
                payload["effective_admission_max_bytes"],
            )

    def test_retention_prunes_old_cache_that_exceeds_retained_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            old = managed / "old-cache"
            self._write_managed_owner(old, last_used=1.0)
            self._write_valid_engine_receipt(old)
            filler = old / "builder-cache.sparse"
            with filler.open("wb") as handle:
                handle.truncate(21 * 1024**3)

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=6,
                max_bytes=20 * 1024**3,
                planned_growth=30 * 1024**3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(old.exists())
            self.assertEqual(
                [row["name"] for row in payload["removed"]],
                ["old-cache"],
            )
            self.assertLessEqual(
                payload["retained_noncurrent_bytes"],
                payload["retained_cache_budget_bytes"],
            )
            self.assertEqual(
                payload["active_working_set_bytes"],
                payload["current_namespace_bytes"] + 30 * 1024**3,
            )

    def test_large_audit_reserve_is_admitted_without_deleting_in_budget_caches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            current = managed / "current"
            self._write_managed_owner(current, last_used=3.0)
            self._write_valid_engine_receipt(current, payload=b"current")
            with (current / "existing-cache.sparse").open("wb") as handle:
                handle.truncate(6 * 1024**3)

            retained_names = [
                "old-a", "old-b", "old-c", "partial-a", "partial-b",
            ]
            for index, name in enumerate(retained_names, start=1):
                namespace = managed / name
                self._write_managed_owner(namespace, last_used=float(index))
                if name.startswith("old-"):
                    self._write_valid_engine_receipt(
                        namespace, payload=name.encode("utf-8"),
                    )

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=6,
                max_bytes=20 * 1024**3,
                planned_growth=30 * 1024**3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(payload["admission_ok"])
            self.assertEqual(payload["removed"], [])
            self.assertEqual(
                payload["retention_fast_path"],
                "existing_stable_namespace_within_limits",
            )
            self.assertGreater(
                payload["planned_current_growth_bytes"],
                payload["retained_cache_budget_bytes"],
            )
            self.assertLessEqual(
                payload["retained_noncurrent_bytes"],
                payload["retained_cache_budget_bytes"],
            )
            self.assertLessEqual(
                payload["projected_managed_bytes"],
                payload["effective_admission_max_bytes"],
            )
            for name in retained_names:
                self.assertTrue((managed / name).exists())

    def test_retention_never_deletes_an_active_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            active = managed / "active-cache"
            self._write_managed_owner(active, last_used=1.0)
            self._write_valid_engine_receipt(active)
            lock_path = active / ".active.lock"
            lock_path.touch()
            with lock_path.open("r+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed, payload = self._run_retention(
                    base,
                    current_key="new-current",
                    max_namespaces=1,
                    max_bytes=1024**3,
                )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            self.assertEqual(completed.returncode, 75, completed.stderr)
            self.assertTrue(active.exists())
            self.assertFalse((managed / "new-current").exists())
            self.assertIn(
                "active_or_unsafe_lock",
                {row["reason"] for row in payload["protected"]},
            )

    def test_late_active_candidate_prevents_all_unhelpful_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            inactive_oldest = managed / "inactive-oldest"
            active_newer = managed / "active-newer"
            self._write_managed_owner(inactive_oldest, last_used=1.0)
            self._write_valid_engine_receipt(inactive_oldest, payload=b"inactive")
            self._write_managed_owner(active_newer, last_used=2.0)
            self._write_valid_engine_receipt(active_newer, payload=b"active")
            active_lock = active_newer / ".active.lock"
            active_lock.touch()

            with active_lock.open("r+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed, payload = self._run_retention(
                    base,
                    current_key="new-current",
                    max_namespaces=1,
                    max_bytes=1024**3,
                )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

            self.assertEqual(completed.returncode, 75, completed.stderr)
            self.assertEqual(payload["removed"], [])
            self.assertTrue(inactive_oldest.exists())
            self.assertTrue(active_newer.exists())
            self.assertFalse((managed / "new-current").exists())

    def test_existing_current_and_planned_growth_form_active_working_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "splitpoint_runs"
            current = (
                base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
                / "current"
            )
            self._write_managed_owner(current, last_used=1.0)
            self._write_valid_engine_receipt(current)
            filler = current / "existing-engine-cache.sparse"
            with filler.open("wb") as handle:
                handle.truncate(12 * 1024**3)

            completed, payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=6,
                max_bytes=20 * 1024**3,
                planned_growth=12 * 1024**3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(current.exists())
            self.assertNotIn("new_empty_current_rolled_back", payload)
            self.assertEqual(
                payload["active_working_set_bytes"],
                payload["current_namespace_bytes"] + 12 * 1024**3,
            )
            self.assertGreater(
                payload["projected_managed_bytes"],
                payload["max_bytes"],
            )
            self.assertLessEqual(
                payload["projected_managed_bytes"],
                payload["effective_admission_max_bytes"],
            )
            self.assertEqual(payload["retained_noncurrent_bytes"], 0)

    def test_retention_lock_symlink_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "splitpoint_runs"
            managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
            managed.mkdir(parents=True)
            outside = root / "outside.lock"
            outside.write_text("sentinel", encoding="utf-8")
            (managed / ".retention.lock").symlink_to(outside)

            completed, _payload = self._run_retention(
                base,
                current_key="current",
                max_namespaces=6,
                max_bytes=1024**3,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
