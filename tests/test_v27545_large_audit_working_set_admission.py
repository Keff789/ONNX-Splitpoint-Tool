from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_trt_cache_retention_command,
)
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import (
    score_independent_audit_start_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PROFILE = (
    ROOT
    / "onnx_splitpoint_tool/resources/evaluation_profiles/"
    "thesis_final_campaign_v1.yaml"
)
GIB = 1024**3


def _import_gui_app_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Import GUI orchestration without switching an existing headless backend."""

    import matplotlib

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    return importlib.import_module("onnx_splitpoint_tool.gui.app")


def _audit_plan(audit_size: int) -> dict[str, Any]:
    return {
        "score_independent_audit_counts": {
            "resnet50": audit_size,
            "yolo26s": audit_size,
            "yolov7_paper": audit_size,
        },
        "execution_union_candidate_count_min_total": 3 * audit_size,
        "execution_union_candidate_count_upper_bound_total": 3 * audit_size,
        "expected_generic_result_rows_min_total": 12 * audit_size,
        "expected_generic_result_rows_total": 12 * audit_size,
        "native_enabled": True,
        "native_energy_enabled": True,
    }


@pytest.mark.parametrize("audit_size", [20, 30])
def test_large_audit_sizes_remain_legal_and_have_an_explicit_start_contract(
    audit_size: int,
) -> None:
    profile = yaml.safe_load(REFERENCE_PROFILE.read_text(encoding="utf-8"))
    profile["selection_policy"].update(
        {
            "selection_strategy": "score_independent_audit",
            "score_independent_audit_enabled": True,
            "audit_size": audit_size,
            "minimum_valid_audit_candidates": min(10, audit_size),
        }
    )

    validated = validate_evaluation_profile_payload(
        profile, source=f"v27545-audit-{audit_size}"
    )
    summary = score_independent_audit_start_summary(_audit_plan(audit_size))

    assert validated["selection_policy"]["audit_size"] == audit_size
    assert summary == {
        "schema": "onnx-splitpoint/gui-audit-start-summary",
        "schema_version": 1,
        "confirmation_required": True,
        "audit_counts": {
            "resnet50": audit_size,
            "yolo26s": audit_size,
            "yolov7_paper": audit_size,
        },
        "execution_union_candidate_count_min_total": 3 * audit_size,
        "execution_union_candidate_count_upper_bound_total": 3 * audit_size,
        "expected_generic_result_rows_min_total": 12 * audit_size,
        "expected_generic_result_rows_total": 12 * audit_size,
        "native_enabled": True,
        "native_energy_enabled": True,
        "concise": summary["concise"],
        "confirmation_text": summary["confirmation_text"],
    }
    assert str(audit_size) in summary["confirmation_text"]
    assert "kein kleiner Standardlauf" in summary["confirmation_text"]


def test_audit_minimum_cannot_exceed_audit_size_but_equal_is_legal() -> None:
    profile = yaml.safe_load(REFERENCE_PROFILE.read_text(encoding="utf-8"))
    profile["selection_policy"].update(
        {
            "selection_strategy": "score_independent_audit",
            "score_independent_audit_enabled": True,
            "audit_size": 20,
            "minimum_valid_audit_candidates": 20,
        }
    )
    validated = validate_evaluation_profile_payload(
        profile, source="v27545-equal-audit-minimum"
    )
    assert validated["selection_policy"]["minimum_valid_audit_candidates"] == 20

    invalid = copy.deepcopy(profile)
    invalid["selection_policy"]["minimum_valid_audit_candidates"] = 21
    with pytest.raises(
        ValueError,
        match=(
            r"minimum_valid_audit_candidates \(21\) must be <= "
            r"selection_policy.audit_size \(20\)"
        ),
    ):
        validate_evaluation_profile_payload(
            invalid, source="v27545-impossible-audit-minimum"
        )


class _RecordedVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = str(value)


class _DeferredThread:
    def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


def _gui_start_fixture(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
    opts = SimpleNamespace(
        profile="audit-profile.yaml",
        out=str(tmp_path / "evaluation-runs"),
        run_id="audit-30",
        resume=False,
        force_stage=[],
        profile_start_snapshot={
            "resolved_profile": {"selection_policy": {}},
            "resolved_selection": {
                "profile_name": "audit-profile",
                "run_mode": "standard",
                "native_enabled": True,
                "energy_enabled": True,
                "models": ["resnet50", "yolo26s", "yolov7_paper"],
            },
            "profile_id": "audit-profile",
            "snapshot_sha256": "sha256:test",
        },
        models_root="",
        skip_benchmarks=False,
        no_remote=False,
    )
    status = _RecordedVariable()
    app = SimpleNamespace(
        _eval_workflow_snapshot_options=(
            lambda *, resume_override=None: opts
        ),
        _eval_workflow_command_preview=lambda _opts: "preview",
        _jobs_register=lambda **_kwargs: None,
        _eval_workflow_text_set=lambda _text: None,
        _background_jobs={},
        var_eval_workflow_status=status,
    )
    return app, opts


def test_declining_a_fresh_audit_creates_no_run_or_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gui_app = _import_gui_app_headless(monkeypatch)

    app, opts = _gui_start_fixture(tmp_path)
    monkeypatch.setattr(gui_app, "build_effective_execution_plan", lambda _: _audit_plan(30))
    monkeypatch.setattr(gui_app.messagebox, "askyesno", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        gui_app,
        "_new_evaluation_workflow_job_id",
        lambda: pytest.fail("job id allocated after declined confirmation"),
    )
    app._jobs_register = lambda **_kwargs: pytest.fail(
        "job registered after declined confirmation"
    )

    result = gui_app.SplitPointAnalyserGUI._queue_evaluation_workflow(app)

    assert result is None
    assert app.var_eval_workflow_status.value == (
        "Start abgebrochen: Ranking-Audit nicht bestätigt."
    )
    assert not Path(opts.out).exists()
    assert app._background_jobs == {}


def test_accepting_a_fresh_audit_registers_the_confirmed_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gui_app = _import_gui_app_headless(monkeypatch)

    app, opts = _gui_start_fixture(tmp_path)
    registered: dict[str, Any] = {}

    def _register(**payload: Any) -> None:
        registered.update(payload)
        app._background_jobs[payload["job_id"]] = SimpleNamespace(
            worker_thread=None
        )

    app._jobs_register = _register
    monkeypatch.setattr(gui_app, "build_effective_execution_plan", lambda _: _audit_plan(30))
    monkeypatch.setattr(gui_app.messagebox, "askyesno", lambda *args, **kwargs: True)
    monkeypatch.setattr(gui_app, "_new_evaluation_workflow_job_id", lambda: "job-audit-30")
    monkeypatch.setattr(gui_app.threading, "Thread", _DeferredThread)

    result = gui_app.SplitPointAnalyserGUI._queue_evaluation_workflow(app)

    assert result == "job-audit-30"
    assert Path(opts.out).is_dir()
    assert registered["job_id"] == result
    assert any(
        "Ranking-audit start confirmation: accepted" in line
        for line in registered["initial_lines"]
    )
    thread = app._background_jobs[result].worker_thread
    assert isinstance(thread, _DeferredThread)
    assert thread.started is True


def _write_owner(namespace: Path, *, last_used: float) -> None:
    namespace.mkdir(parents=True, exist_ok=True)
    (namespace / ".splitpoint_trt_cache_owner.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/managed-trt-cache-owner",
                "schema_version": 1,
                "owner": "onnx-splitpoint-tool",
                "cache_key": namespace.name,
                "created_at_unix": last_used,
                "last_used_at_unix": last_used,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _write_receipt(namespace: Path, *, payload: bytes) -> None:
    engine = namespace / "b001" / "engine.plan"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(payload)
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(payload).hexdigest(),
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    (engine.parent / "engine_build_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _retention(
    base: Path, *, current_key: str, planned_growth: int
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    command = _remote_trt_cache_retention_command(
        remote_base=str(base),
        current_key=current_key,
        max_namespaces=6,
        max_bytes=20 * GIB,
        planned_current_growth_bytes=planned_growth,
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    marker = "SPLITPOINT_TRT_RETENTION_JSON="
    payload = next(
        json.loads(line[len(marker) :])
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    )
    return completed, payload


def test_oversized_active_working_set_keeps_reusable_and_protected_caches(
    tmp_path: Path,
) -> None:
    base = tmp_path / "splitpoint-runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    current = managed / "current-audit"
    retained = managed / "retained-reusable"
    active = managed / "active-other-run"
    foreign = managed / "foreign-unowned"

    for index, namespace in enumerate((current, retained, active), start=1):
        _write_owner(namespace, last_used=float(index))
        _write_receipt(namespace, payload=namespace.name.encode("utf-8"))
    with (current / "existing-cache.sparse").open("wb") as handle:
        handle.truncate(6 * GIB)
    foreign.mkdir(parents=True)
    foreign_sentinel = foreign / "do-not-delete.txt"
    foreign_sentinel.write_text("foreign", encoding="utf-8")
    retained_receipt = (retained / "b001" / "engine_build_receipt.json").read_bytes()
    active_receipt = (active / "b001" / "engine_build_receipt.json").read_bytes()
    active_lock = active / ".active.lock"
    active_lock.touch()

    with active_lock.open("r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        first, first_payload = _retention(
            base, current_key=current.name, planned_growth=30 * GIB
        )
        second, second_payload = _retention(
            base, current_key=current.name, planned_growth=30 * GIB
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    for completed, payload in (
        (first, first_payload),
        (second, second_payload),
    ):
        assert completed.returncode == 0, completed.stderr
        assert payload["admission_ok"] is True
        assert payload["removed"] == []
        assert payload["selected_plan_reserve_applied"] is True
        assert payload["planned_current_growth_bytes"] == 30 * GIB
        assert payload["retained_cache_budget_bytes"] == 20 * GIB
        assert payload["active_working_set_bytes"] == (
            payload["current_namespace_bytes"] + 30 * GIB
        )
        assert payload["retained_noncurrent_bytes"] <= 20 * GIB
        assert payload["projected_managed_bytes"] <= (
            payload["effective_admission_max_bytes"]
        )

    assert current.is_dir()
    assert retained.is_dir()
    assert active.is_dir()
    assert foreign.is_dir()
    assert foreign_sentinel.read_text(encoding="utf-8") == "foreign"
    assert (retained / "b001" / "engine_build_receipt.json").read_bytes() == (
        retained_receipt
    )
    assert (active / "b001" / "engine_build_receipt.json").read_bytes() == (
        active_receipt
    )
