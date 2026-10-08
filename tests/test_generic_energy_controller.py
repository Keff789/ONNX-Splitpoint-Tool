import copy
import json
from pathlib import Path
import shutil
import pytest
from onnx_splitpoint_tool.workflow import generic_completed_task as workflow
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal
from tests.test_generic_energy_generated_processes import generated_case


def test_energy_download_preserves_marker_bound_bytes(tmp_path):
    remote=tmp_path/'remote'/'nonce';remote.mkdir(parents=True)
    original=b'{"repetitions": [],  "schema_version": 2}\n'
    (remote/'generic_energy_results.json').write_bytes(original)
    class Transport:
        def scp_download(self,source,destination,**kw):
            shutil.copytree(source,Path(destination)/Path(source).name);return 0,''
    output=tmp_path/'local';output.mkdir()
    result=workflow.download_case(Transport(),str(remote),output,1,result_filename='generic_energy_results.json')
    assert result['result_collected'] is True
    assert (output/'generic_energy_results.json').read_bytes()==original
    assert not (output/'generic_completion_results.json').exists()


def test_capture_requires_normal_owner_before_transport(tmp_path,monkeypatch):
    monkeypatch.delenv('ONNX_SPLITPOINT_REMOTE_PROCESS_LEASE_FILE',raising=False)
    monkeypatch.delenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES',raising=False)
    class Transport:
        def __getattr__(self,name):raise AssertionError('No transport before owner')
    with pytest.raises((RuntimeError,ValueError)):
        workflow.run_energy_case(tmp_path/'absent.json',{},Transport())


def test_durable_start_blocks_implicit_retry(generated_case,monkeypatch,tmp_path):
    plan,descriptors,case=copy.deepcopy(generated_case)
    monkeypatch.setattr(RemoteProcessLeaseJournal,'from_environment',classmethod(lambda cls,**kw:object()))
    monkeypatch.setenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES','required')
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'capture_1_started.json').write_text('{}')
    monkeypatch.setattr(workflow,'stage_energy_case',lambda *a:pytest.fail('No staging after prior attempt'))
    with pytest.raises(FileExistsError,match='already attempted'):workflow.run_energy_case(path,case,None)
