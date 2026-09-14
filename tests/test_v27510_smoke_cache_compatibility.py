from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy,
)
from onnx_splitpoint_tool.runners.native_split_quality_runtime import (
    _cache_binding_identity_mismatch_axes,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE_BINDING_SHA256 = (
    "a07c35d0f3c23432e93d37751fe58dda930bd0674c85bde3a3c51d7aaa8f2542"
)
HISTORICAL_ARTIFACT_SET_SHA256 = (
    "f1cbf4f2d5c5fce0fc328059b1202fbbf86d9fbb2deb70bf53c2e922e2552c44"
)


def _historical_artifacts() -> dict[str, dict[str, Any]]:
    """Identity captured from the successful 2026-08-04 Smoke binding."""

    return {
        "boundary_metadata": {
            "sha256": "d5975b4a293929c38c87ea6da50a6546d4c7ecee1935d0df48e2406000566903",
            "size_bytes": 823,
        },
        "build_part2_onnx": {
            "sha256": "03683ba3a0cd3ebd88f92299591abdd517f8d348e523c058b6596965064e3f0a",
            "size_bytes": 96481563,
        },
        "engine": {
            "sha256": "7b3059984803b5fc4f5a23d9702f17f70739d3067b63c507bae3f78be9307da9",
            "size_bytes": 48621324,
        },
        "engine_build_receipt": {
            "sha256": "cbffb531448e1037111592da70791750c7ea1ab47993dd2e4c15ca0632947b3a",
            "size_bytes": 2186,
        },
        "native_trt_meta": {
            "sha256": "9aa56d8c9893363113e5ba598ec4417e941c40f4af2af16188e4ed680dd3056e",
            "size_bytes": 21928,
        },
        "part1_runtime": {
            "sha256": "79bf798acd0b7adc62f998e351acdf72db3f077dac8da5303576b783f5d0af44",
            "size_bytes": 3522957,
        },
        "source_part2_onnx": {
            "sha256": "03d473fae2a5be694f6ce66202252e56b11b62bee1152105799bda6e7b09d271",
            "size_bytes": 96481222,
        },
        "trtexec": {
            "sha256": "b800990508793a9bf50713d78d4fedfe475381d8cd749abda80909b9197fa1c4",
            "size_bytes": 1695696,
        },
    }


def _load_materializer():
    path = ROOT / "scripts/materialize_cache_verify_native_split_binding.py"
    name = f"v27510_materializer_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _binding(
    *, source_binding_sha256: str = HISTORICAL_SOURCE_BINDING_SHA256,
    engine_sha256: str | None = None,
) -> dict[str, Any]:
    artifacts = _historical_artifacts()
    if engine_sha256 is not None:
        artifacts["engine"]["sha256"] = engine_sha256
    return {
        "binding_sha256": "b" * 64,
        "cache_verify_replay": {
            "artifact_policy": "cache_verify_only",
            "compiler_dispatched": False,
            "source_binding_sha256": source_binding_sha256,
            "local_validation_status": "local_files_rehashed_test",
        },
        "preselection": {"precision": "float32_layout_fp16"},
        "artifacts": artifacts,
    }


def _replay_result(
    module: Any, binding: dict[str, Any], cache_root: Path,
) -> dict[str, Any]:
    source_binding_sha256 = binding["cache_verify_replay"][
        "source_binding_sha256"
    ]
    artifact_set_sha256 = module._artifact_set_sha256(binding)
    persistent_binding_path = str(
        cache_root / "native_split_quality/source/native_split_quality_binding.json"
    )
    return {
        "binding": binding,
        "persistent_binding_path": persistent_binding_path,
        "cache_verify_source_binding_sha256": source_binding_sha256,
        "cache_verify_source_artifact_set_sha256": artifact_set_sha256,
        "cache_verify_equivalence_key_sha256": module.canonical_json_sha256({
            "binding_sha256": source_binding_sha256,
            "artifact_set_sha256": artifact_set_sha256,
        }),
        "cache_verify_exact_binding_count": 1,
        "cache_verify_equivalent_binding_paths": [persistent_binding_path],
    }


def _layout(tmp_path: Path, names: tuple[str, ...]) -> tuple[Path, Path]:
    benchmark_set = tmp_path / "benchmark_set"
    (benchmark_set / "b052").mkdir(parents=True)
    cache_parent = tmp_path / "cache"
    for name in names:
        (cache_parent / name / "native_split_quality").mkdir(parents=True)
    return benchmark_set, cache_parent


def _call(module: Any, benchmark_set: Path, cache_parent: Path, output: Path):
    return module.materialize(
        benchmark_set=benchmark_set,
        model_id="resnet50",
        case_id="b052",
        setup_id="orin_nx_hailo8_01",
        backend="hailo8",
        eval_run_id="cache-verify-v27510",
        engine_cache_root=cache_parent,
        output=output,
    )


def _accept_replay_validation(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(
        module,
        "validate_native_split_quality_binding",
        lambda binding, **_kwargs: (binding, "local_files_rehashed_test"),
    )


def test_packaged_smoke_profile_and_materializer_mirrors_are_byte_identical() -> None:
    assert (
        ROOT / "profiles/cache_verify_resnet50_b052_hailo8.yaml"
    ).read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles"
        / "cache_verify_resnet50_b052_hailo8.yaml"
    ).read_bytes()
    assert (
        ROOT / "scripts/materialize_cache_verify_native_split_binding.py"
    ).read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts"
        / "materialize_cache_verify_native_split_binding.py"
    ).read_bytes()
    policy = known_native_split_policy(
        model_id="resnet50",
        case_id="b052",
        setup_id="orin_nx_hailo8_01",
        backend="hailo8_to_trt",
    )
    assert policy is not None
    module = _load_materializer()
    assert not hasattr(module, "_HISTORICAL_SOURCE_BINDING_SHA256")
    assert not hasattr(module, "_HISTORICAL_ARTIFACT_SET_SHA256")
    assert module._artifact_set_sha256({
        "artifacts": _historical_artifacts(),
    }) == HISTORICAL_ARTIFACT_SET_SHA256
    # Captured from the successful v2.75.x Smoke run on 2026-08-04.
    historical_part1_sha = (
        "79bf798acd0b7adc62f998e351acdf72"
        "db3f077dac8da5303576b783f5d0af44"
    )
    historical_part2_sha = (
        "03d473fae2a5be694f6ce66202252e56"
        "b11b62bee1152105799bda6e7b09d271"
    )
    historical_policy_sha = (
        "0e8091c85bf3a5962e1dbaebd10a566"
        "a3f000b10a7ff31760e9c252fc1b5f778"
    )
    assert policy["policy_sha256"] == historical_policy_sha
    assert _cache_binding_identity_mismatch_axes(
        {
            "artifacts": {
                "part1_runtime": {"sha256": historical_part1_sha},
                "source_part2_onnx": {"sha256": historical_part2_sha},
            },
            "preselection": {"policy_sha256": historical_policy_sha},
        },
        part1_sha256=historical_part1_sha,
        source_part2_sha256=historical_part2_sha,
        policy_sha256=policy["policy_sha256"],
    ) == set()


def test_captured_august_smoke_binding_reconstructs_frozen_identity() -> None:
    module = _load_materializer()
    binding = json.loads((
        ROOT / "tests/fixtures/v27510_cache_canary"
        / "resnet50_b052_hailo8_outer_binding.json"
    ).read_text(encoding="utf-8"))

    assert binding["binding_sha256"] == (
        "e887015bad0ddb943556c961124237ad9f59c0c283a20084a8c5099edd8eeec3"
    )
    assert binding["producer_binding_sha256"] == (
        HISTORICAL_SOURCE_BINDING_SHA256
    )
    assert module._artifact_set_sha256(binding) == (
        HISTORICAL_ARTIFACT_SET_SHA256
    )
    assert binding["artifacts"]["engine"]["sha256"] == (
        "7b3059984803b5fc4f5a23d9702f17f70739d3067b63c507bae3f78be9307da9"
    )


def test_candidate_roots_include_direct_parent_and_historical_suite_roots(
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    cache_parent = tmp_path / "cache"
    (cache_parent / "native_split_quality").mkdir(parents=True)
    (cache_parent / "resnet50-old/native_split_quality").mkdir(parents=True)

    assert module._candidate_cache_roots(cache_parent) == [
        cache_parent.resolve(),
        (cache_parent / "resnet50-old").resolve(),
    ]


@pytest.mark.parametrize(
    "names",
    (
        ("a-old-standard", "b-existing-v275-smoke"),
        ("a-existing-v275-smoke", "b-new-standard"),
    ),
)
def test_all_roots_are_scanned_until_one_exact_historical_smoke_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    names: tuple[str, str],
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(tmp_path, names)
    smoke_name = next(name for name in names if "smoke" in name)
    calls: list[str] = []

    def replay(*, cache_root: Path, **_kwargs: Any) -> dict[str, Any]:
        calls.append(cache_root.name)
        if cache_root.name != smoke_name:
            raise RuntimeError(
                "cache_miss_blocked:artifact_policy=cache_verify_only:"
                "compiler=trtexec:mismatch_axes=part1_runtime_sha256:"
                "artifact=native_split_quality_binding:"
                "reason=exact_hit_count_0"
            )
        return _replay_result(module, _binding(), cache_root)

    monkeypatch.setattr(module, "prepare_native_split_quality_binding", replay)
    _accept_replay_validation(monkeypatch, module)
    output = tmp_path / "out/binding-set.json"

    result = _call(module, benchmark_set, cache_parent, output)

    assert calls == list(names)
    attestation = result["cache_verify_attestation"]
    assert Path(attestation["selected_cache_root"]).name == smoke_name
    assert [row["status"] for row in attestation["cache_root_diagnostics"]] == [
        "exact_hit" if name == smoke_name else "no_exact_hit"
        for name in names
    ]
    assert [
        row["mismatch_axes"]
        for row in attestation["cache_root_diagnostics"]
    ] == [
        [] if name == smoke_name else ["part1_runtime_sha256"]
        for name in names
    ]
    assert output.is_file()


def test_byte_identical_exact_roots_are_deduplicated_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(
        tmp_path, ("a-smoke-copy", "b-smoke-copy"),
    )
    monkeypatch.setattr(
        module,
        "prepare_native_split_quality_binding",
        lambda **kwargs: _replay_result(
            module, _binding(), Path(kwargs["cache_root"]),
        ),
    )
    _accept_replay_validation(monkeypatch, module)
    output = tmp_path / "out/binding-set.json"

    result = _call(module, benchmark_set, cache_parent, output)

    attestation = result["cache_verify_attestation"]
    assert Path(attestation["selected_cache_root"]).name == "a-smoke-copy"
    assert attestation["exact_hit_count"] == 2
    assert attestation["distinct_source_binding_sha256_count"] == 1
    assert attestation["distinct_equivalence_key_count"] == 1
    assert attestation["hashes_are_diagnostic_only"] is True
    assert attestation["compatible_hit_count"] == 2
    assert [
        row["selected"] for row in attestation["cache_root_diagnostics"]
    ] == [True, False]
    assert [
        Path(path).name for path in attestation["equivalent_cache_roots"]
    ] == ["a-smoke-copy", "b-smoke-copy"]
    assert output.exists()


def test_inner_binding_path_outside_candidate_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(tmp_path, ("a-smoke",))

    def replay(**kwargs: Any) -> dict[str, Any]:
        result = _replay_result(
            module, _binding(), Path(kwargs["cache_root"]),
        )
        forged = "/tmp/unrelated-root/forged-binding.json"
        result["persistent_binding_path"] = forged
        result["cache_verify_equivalent_binding_paths"] = [forged]
        return result

    monkeypatch.setattr(
        module, "prepare_native_split_quality_binding", replay,
    )
    _accept_replay_validation(monkeypatch, module)

    with pytest.raises(
        RuntimeError,
        match="inner_replay_cross_link_invalid",
    ):
        _call(
            module, benchmark_set, cache_parent,
            tmp_path / "out/binding-set.json",
        )


def test_semantically_compatible_roots_select_one_complete_set_lexically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(
        tmp_path, ("a-smoke", "b-distinct"),
    )

    def replay(*, cache_root: Path, **_kwargs: Any) -> dict[str, Any]:
        if cache_root.name == "a-smoke":
            binding = _binding()
        else:
            binding = _binding(source_binding_sha256="c" * 64)
        return _replay_result(module, binding, cache_root)

    monkeypatch.setattr(
        module, "prepare_native_split_quality_binding", replay,
    )
    _accept_replay_validation(monkeypatch, module)
    output = tmp_path / "out/binding-set.json"

    result = _call(module, benchmark_set, cache_parent, output)

    attestation = result["cache_verify_attestation"]
    assert Path(attestation["selected_cache_root"]).name == "a-smoke"
    assert attestation["exact_hit_count"] == 2
    assert attestation["compatible_hit_count"] == 2
    assert attestation["distinct_source_binding_sha256_count"] == 2
    assert [
        row["status"] for row in attestation["cache_root_diagnostics"]
    ] == ["exact_hit", "exact_hit"]
    assert attestation["selection_rule"] == (
        "lexicographic_semantically_compatible_complete_artifact_set"
    )
    binding = result["bindings_by_model_case_backend"][
        "resnet50|b052|hailo8_to_trt"
    ]
    assert binding == _binding()
    assert output.exists()


def test_distinct_compatible_artifact_sets_are_never_mixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(
        tmp_path, ("a-smoke", "b-inconsistent"),
    )

    def replay(*, cache_root: Path, **_kwargs: Any) -> dict[str, Any]:
        binding = _binding(
            engine_sha256=(
                "d" * 64 if cache_root.name == "b-inconsistent"
                else "e" * 64
            )
        )
        return _replay_result(module, binding, cache_root)

    monkeypatch.setattr(
        module, "prepare_native_split_quality_binding", replay,
    )
    _accept_replay_validation(monkeypatch, module)
    output = tmp_path / "out/binding-set.json"

    result = _call(module, benchmark_set, cache_parent, output)

    attestation = result["cache_verify_attestation"]
    assert Path(attestation["selected_cache_root"]).name == "a-smoke"
    assert attestation["distinct_equivalence_key_count"] == 2
    assert attestation["compatible_hit_count"] == 2
    selected = result["bindings_by_model_case_backend"][
        "resnet50|b052|hailo8_to_trt"
    ]
    assert selected["artifacts"]["engine"]["sha256"] == "e" * 64
    assert output.exists()


def test_invalid_binding_is_not_skipped_in_favour_of_later_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    benchmark_set, cache_parent = _layout(
        tmp_path, ("a-tampered", "b-valid-smoke"),
    )
    calls: list[str] = []

    def replay(*, cache_root: Path, **_kwargs: Any) -> dict[str, Any]:
        calls.append(cache_root.name)
        if cache_root.name == "a-tampered":
            raise RuntimeError(
                "cache_miss_blocked:artifact_policy=cache_verify_only:"
                "compiler=trtexec:artifact=native_split_quality_binding:"
                "reason=invalid_binding_count_1"
            )
        return _replay_result(module, _binding(), cache_root)

    monkeypatch.setattr(module, "prepare_native_split_quality_binding", replay)
    _accept_replay_validation(monkeypatch, module)

    with pytest.raises(RuntimeError, match="invalid_binding_count_1"):
        _call(
            module, benchmark_set, cache_parent,
            tmp_path / "out/binding-set.json",
        )
    assert calls == ["a-tampered"]


@pytest.mark.parametrize(
    "message",
    (
        "reason=exact_hit_count_0",
        "prefix:reason=missing:extra",
        "reason=exact_hit_count_1",
        "reason=exact_hit_count_2",
    ),
)
def test_only_complete_zero_hit_and_missing_messages_are_skippable(
    message: str,
) -> None:
    module = _load_materializer()
    assert module._per_root_exact_miss(message) == ""
