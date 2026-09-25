from pathlib import Path

from onnx_splitpoint_tool.release_identity import VERSION, BUILD_ID


def test_current_source_release_identity_is_consistent():
    root = Path(__file__).resolve().parents[1]
    assert VERSION == "2.91.0" and BUILD_ID == f"v{VERSION}"
    assert f'version = "{VERSION}"' in (root/'pyproject.toml').read_text()
    assert f'name = "onnx-splitpoint-tool"\nversion = "{VERSION}"' in (root/'uv.lock').read_text()
    updater = (root/'scripts/update_source_release.sh').read_text()
    assert f'EXPECTED_VERSION = "{VERSION}"' in updater
    assert f'EXPECTED_BUILD_ID = "{BUILD_ID}"' in updater
    assert f'--expected-version {VERSION}' in updater
    assert f'ONNX-Splitpoint-Tool_v{VERSION}' in updater
    assert (root/'README.md').read_text().startswith(f'# ONNX Splitpoint Tool v{VERSION}\n')


def test_campaign_script_mirrors_match():
    root = Path(__file__).resolve().parents[1]
    for name in ('energy_measurement_cli.py', 'native_producer_energy_plan.py',
                 'run_native_producer_energy_from_summary.py', 'run_evalrun_native_producer_variants.py'):
        assert (root/'scripts'/name).read_bytes() == (root/'onnx_splitpoint_tool/resources/remote_scripts'/name).read_bytes()


def test_current_hardware_independent_smoke(capsys):
    from onnx_splitpoint_tool import v279_smoke
    assert v279_smoke.main() == 0
    assert f'PASS v{VERSION} smoke' in capsys.readouterr().out


def test_current_smoke_rejects_wrong_version_and_build(monkeypatch):
    from onnx_splitpoint_tool import v282_smoke
    for name, wrong in [('VERSION', '2.82'), ('BUILD_ID', 'v2.83-wrong-build')]:
        with monkeypatch.context() as context:
            context.setattr(v282_smoke, name, wrong)
            assert v282_smoke.main() != 0
