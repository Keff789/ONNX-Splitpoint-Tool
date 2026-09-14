"""Keep the small service records that explain strict preflight blockers."""
import hashlib
import json
import zipfile

from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack


def test_hailo_service_records_are_archived_exactly_without_suite_binaries(tmp_path):
    run = tmp_path / 'failed_run'
    run.mkdir()
    (run / 'run_manifest.json').write_text(json.dumps({
        'schema': 'onnx-splitpoint/evaluation-run-manifest', 'schema_version': 1,
        'run_id': run.name, 'status': 'failed', 'tool_version': '2.80.3',
    }))
    (run / 'profile.yaml').write_text('model_suite: {primary: []}\n')
    (run / 'evaluation_workflow.log').write_text('artifact_cache_preflight blocked\n')
    expected = {}
    excluded = {}
    for model in ('mobilenet_v3_large', 'yolo11l'):
        for name in ('hailo_artifact_service_plan.json', 'hailo_build_service_status.json'):
            relative = f'models/{model}/benchmark_set/{name}'
            expected[relative] = json.dumps({'model_id': model, 'status': 'pending_dfc_build'}).encode()
            excluded[f'models/{model}/benchmark_set/legacy_suite/{name}'] = b'{}'
        excluded[f'models/{model}/benchmark_set/legacy_suite/compiled.hef'] = b'private binary'
    for relative, content in {**expected, **excluded}.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    archive = tmp_path / 'debug_pack.zip'
    result = create_evaluation_debug_pack(run, archive)
    assert result['archive_verification'] == 'verified'
    with zipfile.ZipFile(archive) as z:
        assert set(expected) <= set(z.namelist())
        assert not set(excluded).intersection(z.namelist())
        manifest = json.loads(z.read('debug_pack_manifest.json'))
        assert manifest['backend_artifact_diagnostics']['archived_members'] == sorted(expected)
        records = {item['path']: item for item in manifest['files']}
        for relative, content in expected.items():
            assert z.read(relative) == content
            assert records[relative]['sha256'] == 'sha256:' + hashlib.sha256(content).hexdigest()
            assert records[relative]['diagnostic_kind'] == 'backend_artifact_diagnostic'
