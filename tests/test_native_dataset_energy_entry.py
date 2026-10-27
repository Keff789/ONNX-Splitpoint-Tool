"""The public energy entry accepts and forwards closed Native-Q inputs."""
import importlib.util
import json
from pathlib import Path
import sys


def test_actual_entry_dry_run_forwards_every_closed_case_before_planning(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / 'scripts/run_native_producer_energy_from_summary.py'
    spec = importlib.util.spec_from_file_location('native_dataset_energy_entry_test', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    summary = tmp_path / 'existing_fps.json'
    summary.write_text(json.dumps({'rows': []}))
    cases = [tmp_path / 'closed_a.json', tmp_path / 'closed_b.json']
    for case in cases:
        case.write_text('{}')
    calls = []

    def planner_only(argv, **kwargs):
        calls.append(argv)
        assert kwargs['label'] == 'energy_plan'
        out = Path(argv[argv.index('--out-dir') + 1])
        out.mkdir(parents=True)
        # Preserve the real empty-cohort failure instead of inventing a
        # successful scientific plan. Only this planner subprocess is mocked.
        (out / 'native_producer_energy_plan.json').write_text(json.dumps({
            'rows': [], 'preflight_status': 'blocked_no_runtime_constructible_rows'}))
        return {'rc': 0}

    monkeypatch.setattr(module, '_run', planner_only)
    argv = [str(script), '--summary', str(summary), '--out-dir', str(tmp_path / 'dry_run'),
            '--dry-run', '--duration-s', '60', '--runs', '3']
    for case in cases:
        argv += ['--native-dataset-quality', str(case)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.main() == 3  # Parsed/forwarded; the actual empty-plan guard.
    assert len(calls) == 1
    forwarded = [calls[0][index + 1] for index, token in enumerate(calls[0])
                 if token == '--native-dataset-quality']
    assert forwarded == list(map(str, cases))
    result = json.loads((tmp_path / 'dry_run/native_producer_energy_results.json').read_text())
    assert result['dry_run'] is True
    assert result['started_measurement_count'] == 0
