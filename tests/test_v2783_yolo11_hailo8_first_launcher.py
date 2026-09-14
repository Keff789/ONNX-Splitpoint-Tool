from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_v2783_yolo11_hailo8_first_gate.sh"
PROFILE = ROOT / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
VERIFIER = ROOT / "scripts/verify_v2783_yolo11_gate_a_output.py"


def _write_fake_tool(tmp_path: Path) -> tuple[Path, Path]:
    tool = tmp_path / "tool"
    scripts = tool / "scripts"
    profiles = tool / "profiles"
    scripts.mkdir(parents=True)
    profiles.mkdir()
    (profiles / "yolo11l_v2783_hailo8_first_b5_gate_a.yaml").write_bytes(
        PROFILE.read_bytes()
    )
    (profiles / "Complet_set.yaml").write_text(
        "name: must-never-run\n", encoding="utf-8"
    )
    (scripts / "verify_v2783_yolo11_gate_a_output.py").write_bytes(
        VERIFIER.read_bytes()
    )
    calls = tmp_path / "calls.jsonl"

    (scripts / "recover_b5_build_evidence.py").write_text(
        """\
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['FAKE_CALLS'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'kind': 'recovery', 'args': args}) + '\\n')
if args[0] == 'create':
    out = pathlib.Path(args[args.index('--out') + 1])
    out.write_text('{}', encoding='utf-8')
    schema = 'onnx-splitpoint/build-evidence-recovery-result/v1'
else:
    schema = 'onnx-splitpoint/build-evidence-index-verification/v1'
print(json.dumps({
    'schema': schema,
    'schema_version': 1,
    'ok': True,
    'status': 'PASS',
    'runtime_evidence_included': False,
}))
""",
        encoding="utf-8",
    )
    (scripts / "run_evaluation_workflow.py").write_text(
        """\
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['FAKE_CALLS'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps({'kind': 'workflow', 'args': args}) + '\\n')
root = pathlib.Path(args[args.index('--out') + 1])
profile = pathlib.Path(args[args.index('--profile') + 1])
sys.path.insert(0, os.environ['FAKE_TESTS_DIR'])
from v2783_gate_a_fixture_builder import (
    write_gate_a_anchor_fixture,
    write_gate_a_budget_fixture,
)
outcome = os.environ['FAKE_OUTCOME']
if outcome == 'CANARY_BUDGET_EXHAUSTED':
    run = write_gate_a_budget_fixture(root, profile)
else:
    run = write_gate_a_anchor_fixture(root, profile)
if outcome not in {'ANCHOR_FOUND', 'CANARY_BUDGET_EXHAUSTED'}:
    receipt = run / 'models' / 'yolo11l' / 'benchmark_set' / 'hailo_feasibility_receipt.json'
    payload = json.loads(receipt.read_text(encoding='utf-8'))
    payload['outcome'] = outcome
    payload['stop_workflow'] = False
    payload['state']['outcome'] = outcome
    payload['state'].pop('anchor_boundary', None)
    receipt.write_text(json.dumps(payload), encoding='utf-8')
raise SystemExit(int(os.environ.get('FAKE_WORKFLOW_RC', '0')))
""",
        encoding="utf-8",
    )
    return tool, calls


def _launcher_env(tmp_path: Path, tool: Path, calls: Path) -> dict[str, str]:
    source_run = tmp_path / "source_run"
    source_run.mkdir()
    (source_run / "evaluation_workflow.log").write_text(
        "historical B5 log\n", encoding="utf-8"
    )
    models = tmp_path / "models"
    models.mkdir()
    output = tmp_path / "canaries"
    output.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "TOOL": str(tool),
            "TOOL_PYTHON": sys.executable,
            "SOURCE_RUN": str(source_run),
            "WORKFLOW_LOG": str(source_run / "evaluation_workflow.log"),
            "EVIDENCE_INDEX": str(tmp_path / "evidence" / "gate_a.json"),
            "PROFILE": "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml",
            "OUTPUT_ROOT": str(output),
            "MODELS_ROOT": str(models),
            "FAKE_CALLS": str(calls),
            "FAKE_TESTS_DIR": str(ROOT / "tests"),
            "PYTHONPATH": str(ROOT),
        }
    )
    return env


def _test_launcher(tmp_path: Path, env: dict[str, str]) -> Path:
    """Bind production constants to isolated fixture paths in a test copy."""

    text = LAUNCHER.read_text(encoding="utf-8")
    text = text.replace(
        "SOURCE_RUN=/home/kmika/Models/EvaluationRuns/"
        "complete_set_v2782_canary_b5_20260827_193658",
        "SOURCE_RUN=" + shlex.quote(env["SOURCE_RUN"]),
    )
    text = text.replace(
        "EVIDENCE_INDEX=/home/kmika/.onnx_splitpoint_tool/build_evidence/"
        "v2783_yolo11_gate_a.json",
        "EVIDENCE_INDEX=" + shlex.quote(env["EVIDENCE_INDEX"]),
    )
    launcher = tmp_path / "run_gate_a_under_test.sh"
    launcher.write_text(text, encoding="utf-8")
    return launcher


def _calls(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_launcher_defaults_and_execution_surface_are_narrow() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "complete_set_v2782_canary_b5_20260827_193658" in text
    assert "v2783_yolo11_gate_a.json" in text
    assert "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml" in text
    assert "dc11012eae5fadbab546d979054df64764d416e632b9b660de0d09558567b283" in text
    assert "/home/kmika/Models/EvaluationRunsCanary" in text
    assert '${PROFILE:-' not in text
    assert '${SOURCE_RUN:-' not in text
    assert '${WORKFLOW_LOG:-' not in text
    assert '${EVIDENCE_INDEX:-' not in text
    for required in (
        "--profile-driven",
        "--require-run-mode standard",
        "--execution-mode generate_benchmarksets",
        "--models-root",
    ):
        assert required in text
    for forbidden in (
        "--execution-mode generate_and_run",
        "--native-producer",
        "--energy",
        "run_fresh_standard_workflow.sh",
    ):
        assert forbidden not in text


def test_launcher_creates_index_and_exits_zero_only_for_anchor(tmp_path: Path) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    env.update({"FAKE_OUTCOME": "ANCHOR_FOUND", "FAKE_WORKFLOW_RC": "0"})
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "GATE_A_OUTCOME=ANCHOR_FOUND" in completed.stdout
    assert "ANCHOR_BOUNDARY=b67" in completed.stdout
    assert "B5_BLOCKED=NO" in completed.stdout
    assert "B5_LAUNCHED=NO" in completed.stdout
    calls = _calls(calls_path)
    assert calls[0]["kind"] == "recovery"
    assert calls[0]["args"][0] == "create"
    workflow = calls[1]
    assert workflow["kind"] == "workflow"
    assert workflow["args"] == [
        "--profile",
        str(tool / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"),
        "--out",
        workflow["args"][3],
        "--profile-driven",
        "--require-run-mode",
        "standard",
        "--execution-mode",
        "generate_benchmarksets",
        "--models-root",
        str(tmp_path / "models"),
    ]


def test_profile_environment_override_is_ignored(tmp_path: Path) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    env.update(
        {
            "PROFILE": str(tool / "profiles/Complet_set.yaml"),
            "FAKE_OUTCOME": "ANCHOR_FOUND",
            "FAKE_WORKFLOW_RC": "0",
        }
    )
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    workflow = _calls(calls_path)[1]
    assert workflow["kind"] == "workflow"
    profile_arg = workflow["args"][workflow["args"].index("--profile") + 1]
    assert profile_arg == str(
        tool / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
    )
    assert profile_arg != env["PROFILE"]


def test_profile_byte_tamper_stops_before_recovery_or_workflow(
    tmp_path: Path,
) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    frozen = tool / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
    frozen.write_bytes(frozen.read_bytes() + b"\n# byte tamper\n")
    env = _launcher_env(tmp_path, tool, calls_path)
    env.update({"FAKE_OUTCOME": "ANCHOR_FOUND", "FAKE_WORKFLOW_RC": "0"})
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "Gate-A profile SHA-256 mismatch" in completed.stderr
    assert not calls_path.exists()


def test_launcher_verifies_existing_index_and_blocks_b5_on_budget(
    tmp_path: Path,
) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    index = Path(env["EVIDENCE_INDEX"])
    index.parent.mkdir()
    index.write_text("{}", encoding="utf-8")
    env.update(
        {
            "FAKE_OUTCOME": "CANARY_BUDGET_EXHAUSTED",
            "FAKE_WORKFLOW_RC": "1",
        }
    )
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.returncode == 3, completed.stderr + completed.stdout
    assert "GATE_A_OUTCOME=CANARY_BUDGET_EXHAUSTED" in completed.stdout
    assert "ANCHOR_FOUND=NO" in completed.stdout
    assert "CANARY_BUDGET_EXHAUSTED=YES" in completed.stdout
    assert "B5_BLOCKED=YES" in completed.stdout
    assert "B5_LAUNCHED=NO" in completed.stdout
    calls = _calls(calls_path)
    assert calls[0]["kind"] == "recovery"
    assert calls[0]["args"][0] == "verify"
    assert calls[1]["kind"] == "workflow"


def test_production_shaped_anchor_accepts_expected_stop_after_rc1(
    tmp_path: Path,
) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    env.update({"FAKE_OUTCOME": "ANCHOR_FOUND", "FAKE_WORKFLOW_RC": "1"})
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "GATE_A_VERIFICATION=PASS" in completed.stdout
    assert "GATE_A_OUTCOME=ANCHOR_FOUND" in completed.stdout
    assert "EXPECTED_STOP_AFTER_PARTIAL=PASS" in completed.stdout
    assert "WORKFLOW_RC=1" in completed.stdout
    assert "B5_BLOCKED=NO" in completed.stdout


def test_evidence_conflict_is_invalid_and_blocks_b5(tmp_path: Path) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    env.update({"FAKE_OUTCOME": "EVIDENCE_CONFLICT", "FAKE_WORKFLOW_RC": "1"})
    launcher = _test_launcher(tmp_path, env)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr + completed.stdout
    assert "GATE_A_OUTCOME=INVALID" in completed.stdout
    assert "receipt_nonterminal_outcome:EVIDENCE_CONFLICT" in completed.stdout
    assert "B5_BLOCKED=YES" in completed.stdout
    assert "B5_BLOCKED=NO" not in completed.stdout
    assert "B5_LAUNCHED=NO" in completed.stdout


def test_symlink_ancestor_is_refused_without_creating_missing_child(
    tmp_path: Path,
) -> None:
    for field in ("OUTPUT_ROOT", "EVIDENCE_INDEX"):
        case = tmp_path / field.lower()
        case.mkdir()
        tool, calls_path = _write_fake_tool(case)
        env = _launcher_env(case, tool, calls_path)
        outside = case / "outside"
        outside.mkdir()
        symlink = case / "redirect"
        symlink.symlink_to(outside, target_is_directory=True)
        missing = outside / "must_not_be_created"
        if field == "OUTPUT_ROOT":
            env[field] = str(symlink / missing.name)
        else:
            env[field] = str(symlink / missing.name / "gate_a.json")
        env.update({"FAKE_OUTCOME": "ANCHOR_FOUND", "FAKE_WORKFLOW_RC": "0"})
        launcher = _test_launcher(case, env)

        completed = subprocess.run(
            ["bash", str(launcher)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

        assert completed.returncode != 0
        assert not missing.exists()
        assert not missing.is_symlink()
        assert not calls_path.exists()


def test_source_and_evidence_environment_overrides_are_ignored(
    tmp_path: Path,
) -> None:
    tool, calls_path = _write_fake_tool(tmp_path)
    env = _launcher_env(tmp_path, tool, calls_path)
    frozen_source = Path(env["SOURCE_RUN"])
    frozen_index = Path(env["EVIDENCE_INDEX"])
    launcher = _test_launcher(tmp_path, env)

    decoy_source = tmp_path / "decoy_source"
    decoy_source.mkdir()
    (decoy_source / "evaluation_workflow.log").write_text(
        "must not be consumed\n", encoding="utf-8"
    )
    decoy_index = tmp_path / "decoy_evidence" / "redirected.json"
    env.update(
        {
            "SOURCE_RUN": str(decoy_source),
            "WORKFLOW_LOG": str(decoy_source / "evaluation_workflow.log"),
            "EVIDENCE_INDEX": str(decoy_index),
            "FAKE_OUTCOME": "ANCHOR_FOUND",
            "FAKE_WORKFLOW_RC": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    recovery = _calls(calls_path)[0]
    assert recovery["args"][recovery["args"].index("--run-dir") + 1] == str(
        frozen_source
    )
    assert recovery["args"][recovery["args"].index("--out") + 1] == str(
        frozen_index
    )
    assert f"BUILD_EVIDENCE_INDEX={frozen_index}" in completed.stdout
    assert str(decoy_index) not in completed.stdout
    assert not decoy_index.exists()
