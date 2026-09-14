"""Original profile -> standard resolver -> exact selective acceptance scope."""
import copy
from pathlib import Path

import yaml
from scripts import reference_workflow_gate_v282 as gate

FIXTURE = Path(__file__).parent / "fixtures/v282_targeted_profile"


def test_targeted_preparation_retains_recipe_and_restricts_case_work(tmp_path):
    original = (FIXTURE / "profile.yaml").read_bytes()
    source = yaml.safe_load(original)
    cold = []
    for group, scope in gate.SCOPE.items():
        path, report = gate.prepare(FIXTURE, tmp_path / group, group, require_originals=False)
        profile = yaml.safe_load(path.read_text())
        assert profile["selection_policy"]["forced_cases"] == scope["models"]
        assert profile["native_producers"]["case_map"] == scope["models"]
        assert profile["hailo_build"]["force_build"] is False
        for key in ("preset", "optimization_level", "calib_count", "calib_batch_size", "cache_integrity"):
            assert profile["hailo_build"][key] == source["hailo_build"][key]
        assert profile["validation_execution"]["max_items"] == {"classification": 500, "detection": 500}
        assert profile["native_producers"]["frames"] == 1000
        assert profile["energy"]["enabled"] is False
        assert profile["native_producers"]["energy"]["enabled"] is False
        assert report["status"] == "prepared_not_executed"
        cold.extend((m, c) for m, cs in scope["allowed_missing"].items() for c in cs)
    assert cold == [("yolo11l", "b064"), ("yolo26m", "b040"), ("yolo26s", "b023")]
    assert (FIXTURE / "profile.yaml").read_bytes() == original


def test_scope_contains_no_regnet_rebuild_or_known_reject_dispatch():
    all_jobs = {(m, c) for scope in gate.SCOPE.values() for m, cs in scope["models"].items() for c in cs}
    assert not any(m == "regnet_x_1_6gf" for m, _ in all_jobs)
    assert ("yolo26m", "b398") not in all_jobs
    assert ("yolo26s", "b364") not in all_jobs
    assert {(m, c) for m, c in all_jobs if m == "yolov7_paper"} == {
        ("yolov7_paper", c) for c in ("b009", "b011", "b044", "b063")}
