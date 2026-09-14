from __future__ import annotations
import argparse, json
from .artifact_store import ArtifactStore, default_artifact_store_root

def _parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Manage the ONNX Split-Point compiler artifact library")
    p.add_argument("--root", default=str(default_artifact_store_root()))
    sub=p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    idx=sub.add_parser("index-existing"); idx.add_argument("roots", nargs="*")
    ls=sub.add_parser("list"); ls.add_argument("--kind",default=""); ls.add_argument("--pinned",action="store_true"); ls.add_argument("--limit",type=int,default=100)
    v=sub.add_parser("verify"); v.add_argument("--strict",action="store_true"); v.add_argument("--quarantine",action="store_true")
    pin=sub.add_parser("pin"); pin.add_argument("artifact_id",type=int); pin.add_argument("--label",default="final")
    unpin=sub.add_parser("unpin"); unpin.add_argument("artifact_id",type=int)
    prune=sub.add_parser("prune"); prune.add_argument("--older-than-days",type=float,default=30); prune.add_argument("--apply",action="store_true")
    exp=sub.add_parser("export"); exp.add_argument("output"); exp.add_argument("--all",action="store_true")
    imp=sub.add_parser("import"); imp.add_argument("input")
    return p

def main(argv:list[str]|None=None)->int:
    ns=_parser().parse_args(argv); store=ArtifactStore(ns.root)
    if ns.command=="status": payload=store.stats()
    elif ns.command=="index-existing": payload=store.index_existing(ns.roots or None)
    elif ns.command=="list": payload=[r.__dict__ for r in store.list(kind=ns.kind,pinned_only=ns.pinned,limit=ns.limit)]
    elif ns.command=="verify": payload=store.verify(strict=ns.strict,quarantine=ns.quarantine)
    elif ns.command=="pin": payload={"updated":store.pin(ns.artifact_id,label=ns.label)}
    elif ns.command=="unpin": payload={"updated":store.unpin(ns.artifact_id)}
    elif ns.command=="prune": payload=store.prune(older_than_days=ns.older_than_days,dry_run=not ns.apply)
    elif ns.command=="export": payload={"path":str(store.export_pack(ns.output,pinned_only=not ns.all))}
    elif ns.command=="import": payload=store.import_pack(ns.input)
    else: raise AssertionError(ns.command)
    print(json.dumps(payload,indent=2,ensure_ascii=False,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
