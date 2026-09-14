#!/usr/bin/env python3
"""Copy per-model BenchmarkSets from an EvaluationRun to a remote NX for Native-FIFO smokes.

v59az:
- create remote destination with `ssh mkdir -p` before rsync;
- avoid duplicate per-model copies by preferring `benchmark_set/legacy_suite` over
  the parent `benchmark_set` when both exist.
"""
from __future__ import annotations
import argparse, json, shlex, subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def model_name(run_dir: Path, bs: Path) -> str:
    try:
        rel=bs.relative_to(run_dir).parts
        if len(rel)>=2 and rel[0]=="models": return rel[1]
    except Exception: pass
    if bs.name == "legacy_suite" and bs.parent.parent.name:
        return bs.parent.parent.name
    if bs.parent.name == "benchmark_set" and bs.parent.parent.name:
        return bs.parent.parent.name
    return bs.name

def _is_legacy_suite(p: Path) -> bool:
    return p.name == "legacy_suite"

def find_benchmark_sets(root: Path) -> List[Path]:
    """Find runnable benchmark set roots, one per model when possible."""
    candidates=[]
    for pat in ["models/*/benchmark_set/legacy_suite", "models/*/benchmark_set"]:
        for p in root.glob(pat):
            if (p/"benchmark_set.json").exists():
                candidates.append(p)
    if (root/"benchmark_set.json").exists():
        candidates.append(root)
    if not candidates:
        for js in root.rglob("benchmark_set.json"):
            s=str(js)
            if "/energy/" in s or "/collector_storage/" in s or "/processed/" in s:
                continue
            candidates.append(js.parent)

    # Deduplicate exact paths.
    tmp=[]; seen=set()
    for p in candidates:
        try: rp=p.resolve()
        except Exception: rp=p
        if rp not in seen:
            seen.add(rp); tmp.append(p)

    # Prefer a model's legacy_suite over the container benchmark_set, because the
    # native FIFO scripts expect the actual runnable suite root with b*/ dirs.
    by_model: Dict[str, Path] = {}
    for p in tmp:
        m=model_name(root, p)
        old=by_model.get(m)
        if old is None:
            by_model[m]=p
        elif _is_legacy_suite(p) and not _is_legacy_suite(old):
            by_model[m]=p
        elif _is_legacy_suite(p) == _is_legacy_suite(old) and len(str(p)) < len(str(old)):
            by_model[m]=p
    return sorted(by_model.values(), key=lambda p: model_name(root, p))

def split_remote(remote: str) -> Tuple[str,int]:
    if remote.count(":") == 1 and not remote.startswith("["):
        host, maybe_port = remote.rsplit(":", 1)
        if maybe_port.isdigit():
            return host, int(maybe_port)
    return remote, 22

def ssh_cmd(remote: str, command: str) -> List[str]:
    host, port = split_remote(remote)
    cmd=["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if port != 22:
        cmd += ["-p", str(port)]
    return cmd + [host, command]

def rsync_cmd(src: Path, remote: str, dst: str, delete: bool=False) -> List[str]:
    host, port = split_remote(remote)
    cmd=["rsync", "-a", "--info=progress2"]
    if delete:
        cmd.append("--delete")
    if port != 22:
        cmd += ["-e", f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"]
    return cmd + [str(src)+"/", f"{host}:{dst}/"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--remote", required=True, help="e.g. nx@192.168.0.104 or nx@192.168.0.104:22")
    ap.add_argument("--remote-root", default="/home/nx/native_fifo_evalsets")
    ap.add_argument("--models", default="", help="optional comma-separated model filter")
    ap.add_argument("--execute", action="store_true", help="actually run mkdir+rsync; otherwise only print commands")
    ap.add_argument("--delete", action="store_true", help="pass --delete to rsync so destination mirrors source")
    ap.add_argument("--continue-on-error", action="store_true", help="continue copying remaining models if one copy fails")
    args=ap.parse_args()
    run_dir=Path(args.run_dir).expanduser().resolve()
    filters={m.strip() for m in args.models.split(',') if m.strip()}
    cmds=[]
    for bs in find_benchmark_sets(run_dir):
        m=model_name(run_dir, bs)
        if filters and m not in filters: continue
        remote_bs=f"{args.remote_root.rstrip('/')}/{run_dir.name}/{m}/benchmark_set"
        mkdir=ssh_cmd(args.remote, "mkdir -p " + shlex.quote(remote_bs))
        rsync=rsync_cmd(bs, args.remote, remote_bs, delete=args.delete)
        cmds.append({"model":m,"benchmark_set":str(bs),"remote_benchmark_set":remote_bs,"mkdir_cmd":mkdir,"cmd":rsync})
    print(json.dumps({"ok": True, "run_dir": str(run_dir), "count": len(cmds), "commands": cmds}, indent=2))
    if args.execute:
        failures=[]
        for c in cmds:
            print("[copy]", c["model"], "->", c["remote_benchmark_set"])
            subprocess.check_call(c["mkdir_cmd"])
            try:
                subprocess.check_call(c["cmd"])
            except subprocess.CalledProcessError as e:
                failures.append({"model": c["model"], "returncode": e.returncode, "cmd": e.cmd})
                print(f"[copy][error] model={c['model']} rc={e.returncode}")
                if not args.continue_on_error:
                    raise
        if failures:
            print(json.dumps({"ok": False, "failures": failures}, indent=2, default=str))
            return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
