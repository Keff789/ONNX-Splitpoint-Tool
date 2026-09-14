#!/usr/bin/env python3
"""Write an energy-measurement command file for native FIFO runs.

Run this on the machine that starts the u.RECS energy measurement, e.g. SM2.
It creates a local command file that usually SSHes into the NX and launches the
native FIFO runner there.  This avoids the common mistake of creating
/tmp/yolov7_b066_native_fifo.sh on the NX while energy_measurement_cli.py reads
it on SM2.
"""
from __future__ import annotations
import argparse, json, shlex, subprocess
from pathlib import Path


def _looks_like_local_home_on_remote(path: str, ssh: str) -> bool:
    # Common mistake: passing --tool-dir ~/ONNX-Splitpoint-Tool on SM2 expands
    # before Python starts to /home/kmika/..., but the remote NX user is usually
    # /home/nx.  Keep this as a warning, not as a hard failure.
    return bool(ssh and path.startswith('/home/kmika/'))


def main() -> int:
    ap = argparse.ArgumentParser(description="Create local command file for native FIFO energy measurement")
    ap.add_argument("--out", default="/tmp/yolov7_b066_native_fifo_energy.sh")
    ap.add_argument("--ssh", default="", help="Remote SSH target, e.g. nx@192.168.0.104. If omitted, command runs locally.")
    ap.add_argument("--tool-dir", default="/home/nx/ONNX-Splitpoint-Tool", help="Tool dir as seen on the execution host. With --ssh this is the REMOTE path; do not pass an unquoted '~' from SM2 unless it is meant for the remote shell.")
    ap.add_argument("--remote-tool-dir", default="", help="Alias for --tool-dir that makes the remote-path semantics explicit.")
    ap.add_argument("--benchmark-set", required=True, help="BenchmarkSet path as seen on the execution host")
    ap.add_argument("--case", default="b066")
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--precision", default="uint8_cast_fp16")
    ap.add_argument("--frames", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--queue-depth", type=int, default=3)
    ap.add_argument("--hailo-format", default="uint8")
    ap.add_argument("--dump-outputs", action="store_true", default=False)
    ap.add_argument("--build-during-measurement", action="store_true", default=False, help="Allow cmake/build inside the measured command. Default is off: use prebuilt native FIFO executable via --no-build.")
    ap.add_argument("--include-build", action="store_true", help="Include cmake configure/build in the measured command. Default is runtime-only (--no-build), which is usually what we want for energy.")
    ap.add_argument("--no-dump-outputs-for-energy", action="store_true", help="Force-disable --dump-outputs in the generated energy command even if --dump-outputs was supplied.")
    ap.add_argument("--remote-env", default="", help="Optional shell prefix evaluated on the execution host before cd, e.g. 'export PYTHONNOUSERSITE=1'.")
    ap.add_argument("--probe", action="store_true", help="Run a cheap remote/local preflight that checks tool-dir and script existence before writing the command file.")
    ns = ap.parse_args()
    if ns.remote_tool_dir:
        ns.tool_dir = ns.remote_tool_dir

    warnings = []
    if _looks_like_local_home_on_remote(ns.tool_dir, ns.ssh):
        warnings.append("--tool-dir looks like the SM2 local home path. With --ssh it must be the path on the remote NX, usually /home/nx/ONNX-Splitpoint-Tool. Use --remote-tool-dir /home/nx/ONNX-Splitpoint-Tool.")

    prefix = (ns.remote_env.strip() + " && ") if ns.remote_env.strip() else ""
    inner = " ".join([
        prefix + "cd", shlex.quote(ns.tool_dir), "&&",
        "python", "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "--benchmark-set", shlex.quote(ns.benchmark_set),
        "--case", shlex.quote(ns.case),
        "--hw-arch", shlex.quote(ns.hw_arch),
        "--precision", shlex.quote(ns.precision),
        "--frames", str(int(ns.frames)),
        "--warmup", str(int(ns.warmup)),
        "--queue-depth", str(int(ns.queue_depth)),
        "--hailo-format", shlex.quote(ns.hailo_format),
        "--dump-outputs" if (ns.dump_outputs and not ns.no_dump_outputs_for_energy) else "",
        "" if ns.include_build else "--no-build",
    ]).strip()
    if ns.ssh:
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ns.ssh]
        cmd = " ".join([shlex.quote(x) for x in ssh_cmd]) + " " + shlex.quote(inner)
        probe_cmd = ssh_cmd + [f"{prefix}test -d {shlex.quote(ns.tool_dir)} && test -f {shlex.quote(ns.tool_dir)}/scripts/native_hailo_trt_fifo_from_benchmarkset.py"]
    else:
        cmd = inner
        probe_cmd = ["bash", "-lc", f"{prefix}test -d {shlex.quote(ns.tool_dir)} && test -f {shlex.quote(ns.tool_dir)}/scripts/native_hailo_trt_fifo_from_benchmarkset.py"]

    probe = None
    if ns.probe:
        pr = subprocess.run(probe_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        probe = {"rc": pr.returncode, "stdout_tail": pr.stdout[-2000:], "stderr_tail": pr.stderr[-2000:], "cmd": probe_cmd}
        if pr.returncode != 0:
            warnings.append("Probe failed: tool directory or native FIFO script was not found on the execution host. The command file is still written for inspection, but energy measurement would fail in the duration probe.")

    out = Path(ns.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n", encoding="utf-8")
    out.chmod(0o755)
    payload = {
        "ok": True if not probe or probe.get("rc") == 0 else False,
        "command_file": str(out),
        "command": cmd,
        "execution_host_tool_dir": ns.tool_dir,
        "build_during_measurement": bool(ns.build_during_measurement),
        "warnings": warnings,
        "probe": probe,
        "note": "Use this local path with scripts/energy_measurement_cli.py --command-file. The file must exist on the energy-measurement host. Default command is runtime-only (--no-build) to avoid measuring CMake/build overhead.",
        "example_measure": "python scripts/energy_measurement_cli.py measure --setup-id orin_nx_hailo8_01 --run-id yolov7_b066_native_fifo --command-file " + shlex.quote(str(out)) + " --inference-count " + str(int(ns.frames)) + " --pipeline-fps 95.3 --timeout 900",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
