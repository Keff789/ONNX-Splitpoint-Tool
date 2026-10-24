"""Direct unittest: own tiny archives/metadata only; no HOME or pytest session."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from onnx_splitpoint_tool.remote import collection_recovery as mod


class ArchiveRelations(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='TEST_ONLY_checkpoint_')
        self.addCleanup(self.temporary.cleanup); self.root=Path(self.temporary.name)

    def archive(self, link='run/a', *, chain=False, omit_target=False, wrong_digest=False, symlink=False):
        data=b'TEST_ONLY exact bytes'; digest=hashlib.sha256(data).hexdigest()
        rows=[{'path':'a','size_bytes':len(data),'sha256':digest},
              {'path':'b','size_bytes':len(data),'sha256':'0'*64 if wrong_digest else digest}]
        if omit_target: rows=rows[1:]
        if chain: rows.append({'path':'c','size_bytes':len(data),'sha256':digest})
        path=self.root/'own.tar.gz'
        with tarfile.open(path,'w:gz') as tf:
            a=tarfile.TarInfo('run/a');a.size=len(data);tf.addfile(a,io.BytesIO(data))
            b=tarfile.TarInfo('run/b');b.type=tarfile.SYMTYPE if symlink else tarfile.LNKTYPE;b.linkname=link;tf.addfile(b)
            if chain:
                c=tarfile.TarInfo('run/c');c.type=tarfile.LNKTYPE;c.linkname='run/b';tf.addfile(c)
            manifest=json.dumps({'schema':'onnx-splitpoint/closed-run-byte-manifest','schema_version':1,
                'remote_run_dir':'/TEST_ONLY/splitpoint_runs/run','files':rows}).encode()
            m=tarfile.TarInfo('collection_manifest.json');m.size=len(manifest);tf.addfile(m,io.BytesIO(manifest))
        return path

    def test_internal_hardlink_to_earlier_regular_member_has_equal_verified_bytes(self):
        path=self.archive();m=mod._verify_archive(path,mod._descriptor(path),'/TEST_ONLY/splitpoint_runs/run')
        self.assertEqual(len(m['files']),2)

    def test_external_noncanonical_chain_cycle_unmanifested_or_wrong_content_links_reject(self):
        variants=[{'link':'/run/a'},{'link':'../a'},{'link':'run/../a'},
                  {'link':'run//a'},{'link':'run/b'},{'link':'run/missing'},
                  {'chain':True},{'omit_target':True},{'wrong_digest':True},{'symlink':True}]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                path=self.archive(**kwargs)
                with self.assertRaises(Exception):mod._verify_archive(path,mod._descriptor(path),'/TEST_ONLY/splitpoint_runs/run')


class ExistingArchiveCheckpoint(unittest.TestCase):
    def setUp(self):
        import pytest
        from tests.test_generic_closed_collection_recovery import case,put
        self.put=put;self.temporary=tempfile.TemporaryDirectory(prefix='TEST_ONLY_checkpoint_api_')
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.monkey=pytest.MonkeyPatch();self.addCleanup(self.monkey.undo)
        factory=getattr(case,'__wrapped__',None) or getattr(case,'_fixture_function')
        self.kw,self.remote,self.originals,self.calls=factory(self.root,self.monkey)
        base=mod.SSHTransport
        setup_id=self.kw['setup_id']
        class Recording(base):
            def run(inner,command,timeout_s):
                rc,text=super(Recording,inner).run(command,timeout_s)
                p=Path(inner.diagnostics_dir)/('ssh_TEST_ONLY_'+str(len(self.calls))+'.json')
                put(p,{'target_id':setup_id,'command':command,'returncode':rc,'output':text,'cleanup_proven':True})
                return rc,text
        self.monkey.setattr(mod,'SSHTransport',Recording)
        first=self.root/'first';self.kw['output_dir']=first
        with patch.object(mod,'_verify_archive',side_effect=RuntimeError('unsafe/special archive member')):
            with self.assertRaises(Exception):mod.recover_generic_closed_collection(**self.kw,apply=True,cleanup=True)
        prior=first/'COLLECTION_RECOVERY_RECEIPT.json';v=json.loads(prior.read_text())
        diagnostics=[p for p in (first/'diagnostics').glob('*.json')]
        collector=next(p for p in diagnostics if 'SPLITPOINT_COLLECT_BEGIN' in json.loads(p.read_text())['command'])
        pack=next(p for p in diagnostics if shlex.split(json.loads(p.read_text())['command'])[0]=='python3'
                  and shlex.split(json.loads(p.read_text())['command'])[4]=='pack')
        identity={k:v[k] for k in ('run_id','run_dir','session_id','setup_id','model_id')}
        argv=['PY','-B','TEST_ONLY/recover_generic_closed_collection.py']
        for flag,key in (('--run-dir','run_dir'),('--session-id','session_id'),('--setup-id','setup_id'),('--model-id','model_id')):
            argv.extend([flag,identity[key]])
        argv.extend(['--output-dir',str(first),'--apply','--cleanup'])
        logger=self.root/'prior_result.json';put(logger,{'returncode':1,'argv':argv})
        self.checkpoint=self.root/'checkpoint.json'
        put(self.checkpoint,{'schema':'onnx-splitpoint/generic-closed-collection-archive-checkpoint','schema_version':1,
            **identity,'endpoint':v['endpoint'],'remote_run_dir':str(self.remote),
            'prior_recovery_receipt':mod._descriptor(prior),'prior_logger_result':mod._descriptor(logger),
            'collector_diagnostic':mod._descriptor(collector),'pack_diagnostic':mod._descriptor(pack),
            'remote_archive':v['remote_archive'],'local_archive':mod._descriptor(first/'original_remote_run.tar.gz')})
        auth=Path(self.kw['authorization_path']);a=json.loads(auth.read_text());a['archive_checkpoint']=mod._descriptor(self.checkpoint)
        self.auth=self.root/'checkpoint_auth.json';put(self.auth,a)
        self.kw.update(authorization_path=self.auth,output_dir=self.root/'second',archive_checkpoint_path=self.checkpoint)

    def test_reuses_existing_archive_without_collect_pack_or_scp_and_preserves_history(self):
        first_count=len(self.calls);old_archive=Path(json.loads(self.checkpoint.read_text())['local_archive']['path'])
        old_sha=mod._descriptor(old_archive);base=mod.SSHTransport
        class ReuseOnly(base):
            def run(inner,command,timeout_s):
                values=shlex.split(command)
                if values[0]!='python3' or values[4] not in ('preflight','cleanup'):
                    raise AssertionError('checkpoint cannot collect or repack')
                return super(ReuseOnly,inner).run(command,timeout_s)
            def scp_download(inner,*_a,**_k):raise AssertionError('checkpoint cannot download')
        self.monkey.setattr(mod,'SSHTransport',ReuseOnly)
        result=mod.recover_generic_closed_collection(**self.kw,apply=True,cleanup=True)
        self.assertEqual(result['status'],'completed');self.assertFalse(self.remote.exists())
        self.assertFalse(result['fresh_collection_performed']);self.assertNotIn('fresh_collection_rc',result)
        self.assertEqual(result['previous_collection_rc'],0)
        self.assertEqual(len(self.calls)-first_count,2)
        self.assertEqual(mod._descriptor(old_archive),old_sha)
        self.assertTrue(all(p.read_bytes()==b for p,b in self.originals.items()))

    def test_wrong_checkpoint_scope_cannot_cleanup(self):
        cp=json.loads(self.checkpoint.read_text());cp['session_id']='b'*32;self.put(self.checkpoint,cp)
        a=json.loads(self.auth.read_text());a['archive_checkpoint']=mod._descriptor(self.checkpoint);self.put(self.auth,a)
        with self.assertRaises(Exception):mod.recover_generic_closed_collection(**self.kw,apply=True,cleanup=True)
        self.assertTrue(self.remote.is_dir())

    def test_prior_nonzero_collection_cannot_be_relabelled_reused_zero(self):
        cp=json.loads(self.checkpoint.read_text());p=Path(cp['collector_diagnostic']['path'])
        d=json.loads(p.read_text());d['returncode']=1;self.put(p,d)
        cp['collector_diagnostic']=mod._descriptor(p);self.put(self.checkpoint,cp)
        a=json.loads(self.auth.read_text());a['archive_checkpoint']=mod._descriptor(self.checkpoint);self.put(self.auth,a)
        calls_before=len(self.calls)
        with self.assertRaises(Exception):mod.recover_generic_closed_collection(**self.kw,apply=True,cleanup=True)
        self.assertEqual(len(self.calls),calls_before);self.assertTrue(self.remote.is_dir())


if __name__=='__main__':unittest.main()
