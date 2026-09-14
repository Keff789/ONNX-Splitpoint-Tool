#!/usr/bin/env python3
"""Stage and invoke the v2.75.20 Native Full import closure on all field nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.remote_runtime_closure import (  # noqa: E402
    native_remote_package_closure,
)
from onnx_splitpoint_tool.workflow.hardware_matrix import (  # noqa: E402
    canon_accelerator,
    matrix_for_runtime,
)
from onnx_splitpoint_tool.workflow.runner import (  # noqa: E402
    _sync_remote_package_asset_v263,
    _sync_remote_script_v60i,
    _verify_remote_module_binding_v263,
)


_REQUIRED_BACKENDS = ("hailo8", "hailo10h", "deepx")


def _backend_key(value: Any) -> str:
    token = canon_accelerator(value)
    if token.startswith("hailo10"):
        return "hailo10h"
    if token == "deepx_m1":
        return "deepx"
    return token


def _remote_environment(runtime: Mapping[str, Any]) -> str:
    value = str(
        runtime.get("env")
        or runtime.get("activate")
        or runtime.get("remote_venv")
        or runtime.get("venv")
        or ""
    ).strip()
    if (
        value
        and "/bin/activate" in value
        and not value.lstrip().startswith(("source ", ". ", "export "))
    ):
        value = "source " + value
    return value


def resolve_required_remote_setups(
    profile_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve exactly one enabled physical setup for every Full producer."""

    grouped: dict[str, list[dict[str, Any]]] = {
        backend: [] for backend in _REQUIRED_BACKENDS
    }
    for raw in matrix_for_runtime(profile_payload):
        row = dict(raw)
        backend = _backend_key(row.get("accelerator"))
        if backend in grouped:
            grouped[backend].append(row)
    errors = {
        backend: [str(row.get("id") or "") for row in rows]
        for backend, rows in grouped.items()
        if len(rows) != 1
    }
    if errors:
        raise RuntimeError(
            "v27520_remote_setup_cardinality_invalid:" + repr(errors)
        )

    producer_cfg = (
        dict(profile_payload.get("native_producers") or {})
        if isinstance(profile_payload.get("native_producers"), Mapping)
        else {}
    )
    default_tool_dir = str(
        producer_cfg.get("remote_tool_dir")
        or "/home/nx/ONNX-Splitpoint-Tool"
    ).strip()
    resolved: list[dict[str, Any]] = []
    configured_remotes = (
        dict(producer_cfg.get("remotes") or {})
        if isinstance(producer_cfg.get("remotes"), Mapping)
        else {}
    )
    for backend in _REQUIRED_BACKENDS:
        target = grouped[backend][0]
        runtime = (
            dict(target.get("runtime") or {})
            if isinstance(target.get("runtime"), Mapping)
            else {}
        )
        remote = (
            dict(target.get("remote") or {})
            if isinstance(target.get("remote"), Mapping)
            else runtime
        )
        aliases = {
            "hailo8": ("hailo8", "hailo8_to_trt"),
            "hailo10h": ("hailo10h", "hailo10", "hailo10h_to_trt"),
            "deepx": ("deepx", "deepx_m1", "deepx_to_trt"),
        }[backend]
        override_value: Any = None
        for alias in aliases:
            if alias in configured_remotes:
                override_value = configured_remotes[alias]
                break
        if isinstance(override_value, str):
            override: dict[str, Any] = {"ssh": override_value}
        elif isinstance(override_value, Mapping):
            override = dict(override_value)
        else:
            override = {}
        if override:
            host = str(override.get("host") or "").strip()
            user = str(override.get("user") or "").strip()
            ssh = str(override.get("ssh") or "").strip()
            if not ssh:
                ssh = f"{user}@{host}" if user else host
            port = int(override.get("port") or 22)
            remote_env = _remote_environment(override)
            remote_tool_dir = str(
                override.get("remote_tool_dir") or default_tool_dir
            ).strip()
            setup_id = str(
                override.get("setup_id") or target.get("id") or ""
            ).strip()
        else:
            host = str(
                remote.get("host") or runtime.get("host") or ""
            ).strip()
            user = str(
                remote.get("user") or runtime.get("user") or ""
            ).strip()
            ssh = f"{user}@{host}" if user else host
            port = int(remote.get("port") or runtime.get("port") or 22)
            remote_env = _remote_environment(runtime)
            remote_tool_dir = str(
                runtime.get("remote_tool_dir")
                or remote.get("remote_tool_dir")
                or default_tool_dir
            ).strip()
            setup_id = str(target.get("id") or "").strip()
        if not ssh:
            raise RuntimeError(f"v27520_remote_host_missing:{backend}")
        if port != 22:
            raise RuntimeError(
                f"v27520_remote_nonstandard_port_not_supported:{backend}:{port}"
            )
        resolved.append({
            "backend": backend,
            "setup_id": setup_id,
            "ssh": ssh,
            "remote_env": remote_env,
            "remote_tool_dir": remote_tool_dir,
        })
    return resolved


def _default_process_runner(
    command: Sequence[str],
    *,
    label: str,
    timeout_s: int,
):
    del label
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )


def run_remote_contract_preflight(
    profile_payload: Mapping[str, Any],
    *,
    timeout: int = 600,
    sync_script: Callable[..., Any] | None = None,
    sync_asset: Callable[..., Any] | None = None,
    verify_module: Callable[..., Any] | None = None,
    process_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Stage exact bytes and execute the harmless promoter probe on 3 nodes."""

    sync_script_fn = sync_script or _sync_remote_script_v60i
    sync_asset_fn = sync_asset or _sync_remote_package_asset_v263
    verify_module_fn = verify_module or _verify_remote_module_binding_v263
    process_runner_fn = process_runner or _default_process_runner
    rows: list[dict[str, Any]] = []

    for setup in resolve_required_remote_setups(profile_payload):
        ssh = setup["ssh"]
        remote_tool_dir = setup["remote_tool_dir"]
        remote_env = setup["remote_env"]
        steps: list[dict[str, Any]] = []
        for script_name, tokens in (
            (
                "native_full_baseline_eval_runner.py",
                (
                    "--remote-contract-preflight",
                    "--setup-id",
                    "--quality-request-binding-set",
                ),
            ),
            (
                "native_full_semantic_dump.py",
                ("--backend", "native_full_outputs_manifest.json"),
            ),
            (
                "native_progress.py",
                ("def stream_command", "class NativeProgressJournal"),
            ),
        ):
            steps.extend(sync_script_fn(
                ssh=ssh,
                remote_tool_dir=remote_tool_dir,
                script_name=script_name,
                required_tokens=tokens,
                timeout=timeout,
            ))

        module_proofs: list[dict[str, Any]] = []
        pending_module_checks = []
        for relative, module_name, tokens in native_remote_package_closure():
            module_steps = sync_asset_fn(
                ssh=ssh,
                remote_tool_dir=remote_tool_dir,
                relative_path=relative,
                required_tokens=tokens,
                timeout=timeout,
            )
            steps.extend(module_steps)
            expected_sha = next(
                (
                    str(step.get("expected_sha256") or "")
                    for step in reversed(module_steps)
                    if isinstance(step, Mapping)
                    and str(step.get("expected_sha256") or "")
                ),
                "",
            )
            if not expected_sha:
                raise RuntimeError(
                    f"v27520_remote_asset_hash_missing:{setup['backend']}:{relative}"
                )
            pending_module_checks.append((module_name, relative, expected_sha))
        # The real validation initializer can import any of the staged modules.
        for module_name, relative, expected_sha in pending_module_checks:
            proof = verify_module_fn(
                ssh=ssh,
                remote_tool_dir=remote_tool_dir,
                remote_env=remote_env,
                module_name=module_name,
                relative_path=relative,
                expected_sha256=expected_sha,
                timeout=min(timeout, 180),
            )
            steps.append(dict(proof))
            module_proofs.append({
                "module": module_name,
                "expected_sha256": expected_sha,
                "verification": dict(proof.get("verification") or {}),
            })

        prefix = (remote_env + " && ") if remote_env else ""
        command_text = (
            prefix
            + f"cd {shlex.quote(remote_tool_dir)} && "
            + f"PYTHONPATH={shlex.quote(remote_tool_dir)}:"
            + "${PYTHONPATH:-} "
            + "python -u scripts/native_full_semantic_dump.py --help "
            + ">/dev/null && "
            + f"PYTHONPATH={shlex.quote(remote_tool_dir)}:"
            + "${PYTHONPATH:-} "
            + "python -u scripts/native_full_baseline_eval_runner.py "
            + "--root . --remote-contract-preflight"
        )
        completed = process_runner_fn(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                ssh,
                command_text,
            ],
            label=f"v27520-remote-contract-preflight:{setup['backend']}",
            timeout_s=min(timeout, 180),
        )
        stdout = str(completed.stdout or "")
        stderr = str(completed.stderr or "")
        proof_payload: dict[str, Any] = {}
        try:
            proof_payload = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            proof_payload = {}
        if (
            completed.returncode != 0
            or proof_payload.get("ok") is not True
            or proof_payload.get("helper_invoked") is not True
            or proof_payload.get("fail_closed_result") != []
        ):
            raise RuntimeError(
                "v27520_remote_contract_preflight_failed:"
                f"{setup['backend']}:rc={completed.returncode}:"
                f"{stderr[-1000:] or stdout[-1000:]}"
            )
        rows.append({
            **setup,
            "ok": True,
            "staged_script_count": 3,
            "staged_module_count": len(module_proofs),
            "module_proofs": module_proofs,
            "helper_proof": proof_payload,
            "step_count": len(steps),
        })

    return {
        "schema": "onnx-splitpoint/v27520-native-remote-contract-preflight",
        "schema_version": 1,
        "ok": len(rows) == 3 and all(row.get("ok") is True for row in rows),
        "required_backends": list(_REQUIRED_BACKENDS),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json-out", default="")
    ns = parser.parse_args()

    profile_path = Path(ns.profile).expanduser().resolve(strict=True)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise SystemExit("Evaluation Profile must be a YAML object")
    result = run_remote_contract_preflight(payload, timeout=max(30, ns.timeout))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if ns.json_out:
        output = Path(ns.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
