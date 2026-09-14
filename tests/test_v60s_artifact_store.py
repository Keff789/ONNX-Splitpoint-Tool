from __future__ import annotations
import json
from pathlib import Path
from onnx_splitpoint_tool.artifact_store import ArtifactStore


def test_register_lookup_materialize_and_exact_contract(tmp_path: Path):
    store=ArtifactStore(tmp_path/'store'); src=tmp_path/'compiled.dxnn'; src.write_bytes(b'compiler-data')
    c1={'model':'a','arch':'hailo8','opt':1}; c2={'model':'a','arch':'hailo10','opt':1}
    r=store.register(source_path=src,kind='deepx_dxnn',contract=c1,metadata={'legacy_cache_key':'k1'},source_run='run1')
    assert store.lookup(kind='deepx_dxnn',contract=c1) is not None
    assert store.lookup(kind='deepx_dxnn',contract=c2) is None
    dst=tmp_path/'out'/'compiled.dxnn'; method=store.materialize(r,dst,reference='test')
    assert dst.read_bytes()==b'compiler-data'; assert method
    assert store.find_by_metadata(kind='deepx_dxnn',key='legacy_cache_key',value='k1') is not None


def test_pin_prune_export_import_and_corruption(tmp_path: Path):
    store=ArtifactStore(tmp_path/'store')
    a=tmp_path/'a.dxnn'; a.write_bytes(b'dxnn-a')
    b=tmp_path/'b.dxnn'; b.write_bytes(b'dxnn-b')
    ra=store.register(source_path=a,kind='deepx_dxnn',contract={'x':1},pin=True,pin_label='final')
    rb=store.register(source_path=b,kind='deepx_dxnn',contract={'x':2})
    assert store.stats()['artifact_count']==2
    preview=store.prune(older_than_days=0,dry_run=True)
    assert any(x['artifact_id']==rb.artifact_id for x in preview['items'])
    assert not any(x['artifact_id']==ra.artifact_id for x in preview['items'])
    pack=store.export_pack(tmp_path/'pack.zip',pinned_only=True)
    imported=ArtifactStore(tmp_path/'imported').import_pack(pack)
    assert imported['imported_count']==1
    Path(rb.object_path).write_bytes(b'bad')
    report=store.verify(strict=True)
    assert not report['ok']


def test_build_lock_serialises(tmp_path: Path):
    store=ArtifactStore(tmp_path/'store')
    with store.build_lock('hef',{'a':1},timeout_s=1):
        assert (store.locks_dir / ('hef-'+store.contract_hash({'a':1})+'.lock')).exists()
