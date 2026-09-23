"""Read-only owner observations for explicit physical-resource recovery.

This is one prerequisite of the existing locked recovery operation. Absence of
a PID is not cleanup evidence, source readiness, or permission to start work.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from ..platform_power import host_config_from_setup
from .ssh_transport import SSHTransport


def _observe_processes(local_owners, *, proc_root="/proc", lease_parent="/tmp"):
    """Self-contained probe, also sent through the existing read-only SSH path.

    The known executable and lease checks come from the bounded acceptance
    probe. Only program basenames and process identities leave this function;
    command lines and environments are never returned.
    """
    import json
    import os
    import pathlib
    import stat

    proc = pathlib.Path(proc_root)
    errors, processes, leases, owners = [], [], [], []
    exact = {
        "trtexec", "urecs-data-collector", "power_calculations", "hailortcli",
        "benchmark_suite.py", "benchmark_case.py", "run_benchmark_suite.py",
        "run_benchmark_suite_from_set.py", "native_deepx_full_energy_hotloop.py",
        "run_evalrun_native_producer_variants.py", "update_evalset_native_producers.py",
        "run_native_producer_energy_from_summary.py", "native_full_baseline_eval_runner.py",
        "native_trt_from_benchmarkset.py", "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_hailo10_trt_e2e_from_benchmarkset.py", "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_split_quality_runtime.py", "run_split_onnxruntime.py",
        "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
        "energy_measurement_cli.py",
        "run_evaluation_workflow.py", "analyse_and_split_gui.py",
    }

    def identity(pid):
        try:
            text = (proc / str(pid) / "stat").read_text()
            if not text.startswith(str(pid) + " (") or ") " not in text:
                raise ValueError("invalid process stat")
            fields = text[text.rfind(")") + 2:].split()
            ticks, state = fields[19], fields[0]
            if not ticks.isdecimal() or int(ticks) <= 0 or len(state) != 1:
                raise ValueError("invalid process identity")
            return ticks, state
        except FileNotFoundError:
            # An incomplete proc entry is not proven absence.
            try:
                (proc / str(pid)).stat()
            except FileNotFoundError:
                return None, "absent"
            except OSError:
                pass
            errors.append("process_identity_unreadable:" + str(pid))
        except (OSError, ValueError, IndexError):
            errors.append("process_identity_unreadable:" + str(pid))
        return None, "unknown"

    # Match process_control._procfs_matches_active_pid_namespace: a host /proc
    # mounted in a nested namespace must not be traversed with local PIDs.
    # No privileged readlink of another user's /proc/<pid>/ns is required.
    try:
        raw = (proc / "self/stat").read_text()
        if int(raw.split("(", 1)[0].strip()) != os.getpid():
            errors.append("procfs_pid_namespace_mismatch")
        mounts = (proc / "self/mountinfo").read_text().splitlines()
        matched = []
        for line in mounts:
            left, right = line.split(" - ", 1)
            fields, tail = left.split(), right.split()
            if fields[4] == str(proc):
                matched.append((fields, tail))
        if len(matched) != 1 or matched[0][1][0] != "proc":
            errors.append("procfs_mount_unproven")
        else:
            fields, tail = matched[0]
            options = set(fields[5].split(",") + tail[2].split(","))
            if any((option.startswith("hidepid=") and option != "hidepid=0")
                   or option.startswith("subset=") for option in options):
                errors.append("procfs_visibility_restricted")
    except (OSError, ValueError, IndexError):
        errors.append("procfs_namespace_visibility_unreadable")

    if not isinstance(local_owners, list) or not local_owners:
        # The remote scan has no recorded local owners. Its caller uses an
        # empty list; validation of required local owners happens at the API.
        local_owners = []
    for owner in local_owners:
        pid = owner.get("pid") if isinstance(owner, dict) else None
        ticks = owner.get("start_time_ticks") if isinstance(owner, dict) else None
        if (type(pid) is not int or pid <= 0 or isinstance(ticks, bool)
                or not str(ticks).isdecimal() or int(ticks) <= 0):
            errors.append("recorded_owner_identity_invalid")
            continue
        actual_ticks, state = identity(pid)
        if state == "absent":
            observation = "absent"
        elif state == "unknown":
            observation = "unknown"
        elif actual_ticks != str(ticks):
            observation = "pid_reused"
        elif state == "Z":
            observation = "zombie"
        else:
            observation = "active"
        owners.append({"pid": pid, "start_time_ticks": str(ticks),
                       "observed_start_time_ticks": actual_ticks, "state": observation})

    try:
        entries = list(proc.iterdir())
    except OSError:
        entries = []
        errors.append("process_directory_unreadable")
    for entry in entries:
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            names = [pathlib.PurePosixPath(arg.decode("utf-8", "replace")).name
                     for arg in argv if arg and b"\n" not in arg and b" " not in arg]
            match = sorted((set(names) & exact) | {
                name for name in names
                if name.startswith("native_") and name.endswith("_energy_hotloop.py")})
            if match:
                ticks, state = identity(int(entry.name))
                if state not in {"Z", "absent"}:
                    processes.append({"pid": int(entry.name), "uid": entry.stat().st_uid,
                                      "start_time_ticks": ticks, "matched_programs": match})
        except FileNotFoundError:
            # A process may have exited during enumeration; a surviving entry
            # with unreadable cmdline instead makes visibility incomplete.
            try:
                entry.stat()
            except FileNotFoundError:
                continue
            except OSError:
                pass
            errors.append("process_visibility_incomplete:" + entry.name)
        except OSError:
            errors.append("process_visibility_denied:" + entry.name)

    count = 0
    try:
        roots = sorted(path for path in pathlib.Path(lease_parent).iterdir()
                       if path.name.startswith("onnx_splitpoint-process-leases-"))
        for root in roots:
            if root.is_symlink() or not root.is_dir():
                errors.append("lease_root_invalid:" + root.name)
                continue

            def walk_error(_exc):
                errors.append("lease_directory_visibility_error")

            for directory, directories, filenames in os.walk(
                    root, followlinks=False, onerror=walk_error):
                for name in directories:
                    if pathlib.Path(directory, name).is_symlink():
                        errors.append("lease_directory_symlink")
                for name in filenames:
                    if not name.endswith(".lease.json"):
                        continue
                    path = pathlib.Path(directory, name)
                    count += 1
                    if count > 4096:
                        raise ValueError("lease visibility limit")
                    if not stat.S_ISREG(path.lstat().st_mode):
                        errors.append("lease_file_not_regular")
                        continue
                    row = json.loads(path.read_text())
                    if (not isinstance(row, dict)
                            or row.get("schema") != "onnx-splitpoint/remote-process-lease"
                            or type(row.get("pid")) is not int or row["pid"] <= 1
                            or type(row.get("pgid")) is not int or row["pgid"] != row["pid"]
                            or not isinstance(row.get("token"), str) or not row["token"]
                            or not isinstance(row.get("operation_id"), str)
                            or path.name != row["operation_id"] + ".lease.json"
                            or isinstance(row.get("start_time_ticks"), bool)
                            or not str(row.get("start_time_ticks")).isdecimal()
                            or int(row["start_time_ticks"]) <= 0):
                        errors.append("lease_identity_invalid")
                        continue
                    ticks, state = identity(row["pid"])
                    if ticks == str(row["start_time_ticks"]) and state != "Z":
                        leases.append({"path": str(path), "pid": row["pid"],
                                       "start_time_ticks": ticks})
                    elif state == "absent" or (state == "Z" and ticks == str(row["start_time_ticks"])):
                        # Existing guardian proof covers adopted descendants;
                        # root disappearance alone cannot close an owned lease.
                        drained = path.with_name(row["operation_id"] + ".drained")
                        try:
                            proven = (stat.S_ISREG(drained.lstat().st_mode)
                                      and drained.read_text(encoding="ascii").strip() == row["token"])
                        except (OSError, ValueError):
                            proven = False
                        if not proven:
                            errors.append("lease_root_exited_without_drain:" + str(path))
                    else:
                        errors.append("lease_identity_unresolved:" + str(path))
    except (OSError, ValueError):
        errors.append("lease_visibility_error")

    busy = bool(processes or leases or any(row["state"] == "active" for row in owners))
    return {"ok": not busy and not errors, "read_only": True,
            "status": "busy" if busy else "unknown" if errors else "no_known_competing_work_seen",
            "recorded_owners": owners, "processes": processes,
            "live_product_leases": leases, "visibility_errors": sorted(set(errors)),
            "exclusive_start_permission": False, "cleanup_proven": False}


def observe_resource_owners(
    setup: Mapping[str, Any], *, local_owners: list[dict], diagnostics_dir: Path,
) -> dict[str, Any]:
    """Observe exact controller predecessors and known product work on the DUT.

    The recovery caller must hold its existing exclusive locks and separately
    validate original cleanup evidence and the new source-readiness evidence.
    No process is stopped and no remote lease, upload, or measurement is started.
    """
    controller = _observe_processes(local_owners)
    if not isinstance(local_owners, list) or not local_owners:
        controller["visibility_errors"].append("recorded_owners_required")
        controller.update(ok=False, status="unknown")
    remote: dict[str, Any] = {"ok": False, "read_only": True, "status": "not_observed"}
    # Do not open another connection after an already blocked local observation.
    if not controller["ok"]:
        return {"ok": False, "controller": controller, "remote": remote}
    try:
        transport = SSHTransport(host_config_from_setup(setup))
        transport.diagnostics_dir = Path(diagnostics_dir)
        script = inspect.getsource(_observe_processes)
        command = ("python3 -B - <<'RESOURCE_OWNER_OBSERVATION_PY'\n"
                   + script + "\nimport json\nprint('RESOURCE_OWNERS=' + "
                   "json.dumps(_observe_processes([]), sort_keys=True))\n"
                   "RESOURCE_OWNER_OBSERVATION_PY")
        rc, output = transport.run_read_only(command, timeout=30,
                                            env={"PYTHONDONTWRITEBYTECODE": "1"})
        lines = [line[len("RESOURCE_OWNERS="):] for line in output.splitlines()
                 if line.startswith("RESOURCE_OWNERS=")]
        if rc != 0 or len(lines) != 1:
            raise ValueError("remote observation incomplete")
        remote = json.loads(lines[0])
        if (not isinstance(remote, dict) or type(remote.get("ok")) is not bool
                or remote.get("read_only") is not True
                or not all(isinstance(remote.get(key), list) for key in (
                    "recorded_owners", "processes", "live_product_leases", "visibility_errors"))):
            raise ValueError("remote observation malformed")
        remote["ok"] = bool(remote["ok"]
                            and remote.get("status") == "no_known_competing_work_seen"
                            and not remote["visibility_errors"] and not remote["processes"]
                            and not remote["live_product_leases"] and not remote["recorded_owners"])
    except Exception as exc:
        remote = {"ok": False, "read_only": True, "status": "unknown",
                  "visibility_errors": ["remote_observation_failed:" + type(exc).__name__]}
    return {"ok": bool(controller["ok"] and remote["ok"]),
            "controller": controller, "remote": remote}
