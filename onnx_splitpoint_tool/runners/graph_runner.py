from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ._types import GraphPlan, GraphRunResult, Status
from .artifacts import write_json
from .backends.base import Backend, PreparedHandle
from .interface_transfer import deserialize_tensors, serialize_tensors
from .harness.base import Harness


class GraphRunner:
    """Orchestrates running a 1-stage or 2-stage graph.

    GraphRunner is responsible for:
    - calling Backend.prepare/run/cleanup
    - 2-stage composition (stage1 -> interface -> stage2)
    - measuring timings
    - writing a stable result JSON (graph_run.json)

    It should never crash the caller: exceptions are captured into
    GraphRunResult (status failed/partial/cancelled).
    """

    def __init__(self, backends: Mapping[str, Backend]):
        self.backends: dict[str, Backend] = dict(backends)

    def run_graph(
        self,
        plan: GraphPlan,
        harness: Harness,
        artifacts_dir: Path,
        cancel: Any | None = None,
    ) -> GraphRunResult:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        result = GraphRunResult(
            schema_version=1,
            status="failed",
            plan=_plan_to_dict(plan),
            metrics={},
            warnings=[],
            errors=[],
        )

        graph_json = artifacts_dir / "graph_run.json"

        prepared: list[tuple[str, Backend, PreparedHandle, float]] = []  # (stage, backend, handle, init_ms)
        stage_metrics: dict[str, dict[str, Any]] = {}

        end_to_end_times: list[float] = []
        iface_ser_times: list[float] = []
        iface_des_times: list[float] = []
        iface_bytes: list[int] = []

        last_outputs: Any = None

        exception: Exception | None = None

        try:
            # Inputs
            inputs0 = harness.make_inputs(plan.sample_cfg)

            # Stage prepare
            if len(plan.stages) not in (1, 2):
                raise ValueError(f"GraphPlan.stages must have length 1 or 2, got {len(plan.stages)}")

            for st in plan.stages:
                if st.backend_name not in self.backends:
                    raise KeyError(f"Unknown backend '{st.backend_name}' for stage '{st.name}'")

                backend = self.backends[st.backend_name]

                t0 = time.perf_counter()
                handle = backend.prepare(st.run_cfg, artifacts_dir / f"stage_{st.name}")
                init_ms = (time.perf_counter() - t0) * 1000.0

                prepared.append((st.name, backend, handle, init_ms))
                stage_metrics[st.name] = {
                    "init_build_ms": init_ms,
                    "runtime_ms": {
                        "warmup_total": 0.0,
                        "measured_runs": [],
                        "measured_mean": None,
                        "measured_std": None,
                    },
                }

            # Warmup
            for _i in range(max(0, int(plan.warmup_runs))):
                if _is_cancelled(cancel):
                    result.status = "cancelled"
                    break
                out = self._run_once(plan, prepared, inputs0, measure_interface=False)
                for st_name, rt_ms in out["stage_runtime_ms"].items():
                    stage_metrics[st_name]["runtime_ms"]["warmup_total"] += rt_ms
                last_outputs = out.get("outputs")

            # Measured
            if result.status != "cancelled":
                for _i in range(max(0, int(plan.measured_runs))):
                    if _is_cancelled(cancel):
                        result.status = "cancelled"
                        break
                    try:
                        t0 = time.perf_counter()
                        out = self._run_once(plan, prepared, inputs0, measure_interface=plan.measure_interface)
                        t1 = time.perf_counter()

                        end_to_end_times.append((t1 - t0) * 1000.0)

                        for st_name, rt_ms in out["stage_runtime_ms"].items():
                            stage_metrics[st_name]["runtime_ms"]["measured_runs"].append(rt_ms)

                        if out.get("interface") is not None:
                            iface = out["interface"]
                            iface_ser_times.append(float(iface["serialize_ms"]))
                            iface_des_times.append(float(iface["deserialize_ms"]))
                            iface_bytes.append(int(iface["bytes"]))

                        last_outputs = out.get("outputs")
                    except Exception as e:
                        exception = e
                        break

            # Finalize stats (even on exception)
            for st_name, m in stage_metrics.items():
                runs = m["runtime_ms"]["measured_runs"]
                if runs:
                    m["runtime_ms"]["measured_mean"] = float(np.mean(runs))
                    m["runtime_ms"]["measured_std"] = float(np.std(runs, ddof=0))

            iface_metrics: dict[str, Any] | None = None
            if len(plan.stages) == 2 and iface_ser_times:
                iface_metrics = {
                    "serialize_ms": {
                        "measured_mean": float(np.mean(iface_ser_times)),
                        "measured_std": float(np.std(iface_ser_times, ddof=0)),
                    },
                    "deserialize_ms": {
                        "measured_mean": float(np.mean(iface_des_times)),
                        "measured_std": float(np.std(iface_des_times, ddof=0)),
                    },
                    "bytes": {
                        "measured_mean": float(np.mean(iface_bytes)),
                        "measured_max": int(np.max(iface_bytes)) if iface_bytes else 0,
                    },
                }

            result.metrics = {
                "stage": stage_metrics,
                "interface": iface_metrics,
                "end_to_end_ms": {
                    "measured_mean": float(np.mean(end_to_end_times)) if end_to_end_times else None,
                    "measured_std": float(np.std(end_to_end_times, ddof=0)) if end_to_end_times else None,
                    "measured_runs": end_to_end_times,
                },
            }

            # Optional postprocess should never crash the run.
            try:
                _ = harness.postprocess(last_outputs, context={"plan": result.plan})
            except Exception as e:
                result.warnings.append(f"Harness.postprocess failed: {type(e).__name__}: {e}")

            # Status
            if result.status == "cancelled":
                # keep cancelled
                pass
            elif exception is None:
                result.status = "ok"
            else:
                # partial if at least one measured run finished
                result.status = "partial" if end_to_end_times else "failed"
                result.errors.append(f"Run failed: {type(exception).__name__}: {exception}")

            return result

        except Exception as e:
            result.status = "failed"
            result.errors.append(f"GraphRunner exception: {type(e).__name__}: {e}")
            return result

        finally:
            # Always cleanup
            for st_name, backend, handle, _init_ms in reversed(prepared):
                try:
                    backend.cleanup(handle)
                except Exception as e:
                    result.warnings.append(
                        f"Backend.cleanup failed for stage '{st_name}' ({backend.name}): {type(e).__name__}: {e}"
                    )

            # Always write artifact
            try:
                write_json(graph_json, result.to_dict())
            except Exception:
                # Do not crash caller on artifact failures.
                pass

    def _run_once(
        self,
        plan: GraphPlan,
        prepared: list[tuple[str, Backend, PreparedHandle, float]],
        inputs0: dict,
        *,
        measure_interface: bool,
    ) -> dict[str, Any]:
        """Run the graph once and return outputs + per-stage runtimes."""

        if len(prepared) == 1:
            st_name, backend, handle, _ = prepared[0]
            t0 = time.perf_counter()
            out = backend.run(handle, inputs0)
            t1 = time.perf_counter()
            return {
                "outputs": out.outputs,
                "stage_runtime_ms": {st_name: (t1 - t0) * 1000.0},
                "stage1_outputs": None,
                "interface": None,
            }

        # Two-stage
        (s1_name, b1, h1, _), (s2_name, b2, h2, _) = prepared

        # Stage 1
        t0 = time.perf_counter()
        out1 = b1.run(h1, inputs0)
        t1 = time.perf_counter()

        stage1_ms = (t1 - t0) * 1000.0

        # Build stage2 inputs: take tensors from stage1 outputs where possible,
        # else pass-through original inputs.
        stage1_map = _to_tensor_map(out1.outputs)
        stage2_input_names = list(h2.input_names)

        needed_from_stage1 = {k: stage1_map[k] for k in stage2_input_names if k in stage1_map}

        iface = None
        stage2_inputs = None

        if measure_interface:
            ts0 = time.perf_counter()
            ser = serialize_tensors(needed_from_stage1)
            ts1 = time.perf_counter()

            td0 = time.perf_counter()
            deser = deserialize_tensors(ser)
            td1 = time.perf_counter()

            iface = {
                "serialize_ms": (ts1 - ts0) * 1000.0,
                "deserialize_ms": (td1 - td0) * 1000.0,
                "bytes": ser.total_bytes,
            }

            stage2_inputs = {k: deser[k] if k in deser else inputs0[k] for k in stage2_input_names}
        else:
            stage2_inputs = {k: stage1_map[k] if k in stage1_map else inputs0[k] for k in stage2_input_names}

        # Stage 2
        t2 = time.perf_counter()
        out2 = b2.run(h2, stage2_inputs)
        t3 = time.perf_counter()

        stage2_ms = (t3 - t2) * 1000.0

        return {
            "outputs": out2.outputs,
            "stage1_outputs": out1.outputs,
            "stage_runtime_ms": {s1_name: stage1_ms, s2_name: stage2_ms},
            "interface": iface,
        }


def _to_tensor_map(outputs: Any) -> dict[str, np.ndarray]:
    if isinstance(outputs, dict):
        return {str(k): np.asarray(v) for k, v in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return {str(i): np.asarray(v) for i, v in enumerate(outputs)}
    raise TypeError(f"Unsupported outputs type: {type(outputs).__name__}")


def _is_cancelled(cancel: Any | None) -> bool:
    if cancel is None:
        return False
    if hasattr(cancel, "is_cancelled"):
        try:
            return bool(cancel.is_cancelled())
        except Exception:
            return False
    if hasattr(cancel, "is_set"):
        try:
            return bool(cancel.is_set())
        except Exception:
            return False
    if hasattr(cancel, "cancelled"):
        try:
            return bool(cancel.cancelled)
        except Exception:
            return False
    return False


def _plan_to_dict(plan: GraphPlan) -> dict[str, Any]:
    return {
        "label": plan.label,
        "warmup_runs": plan.warmup_runs,
        "measured_runs": plan.measured_runs,
        "measure_interface": plan.measure_interface,
        "sample_cfg": asdict(plan.sample_cfg),
        "stages": [
            {
                "name": s.name,
                "backend": s.backend_name,
                "model_path": str(s.run_cfg.model_path),
                "options": s.run_cfg.options,
            }
            for s in plan.stages
        ],
    }
