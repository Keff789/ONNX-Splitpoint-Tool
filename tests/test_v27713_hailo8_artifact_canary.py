from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v27713_hailo8_artifact_canary.py"
VERIFIER = ROOT / "scripts/verify_v27713_hailo8_canary_payload.py"


def _load_module():
    name = "v27713_hailo8_artifact_canary"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_verifier():
    name = "v27713_hailo8_canary_payload_verifier"
    spec = importlib.util.spec_from_file_location(name, VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _benchmark_set(
    tmp_path: Path, *, image_count: int = 2, model_id: str = "yolo26m",
) -> Path:
    root = tmp_path / "model" / "benchmark_set" / "legacy_suite"
    hef = root / "hailo/hailo8/full/compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"real-precompiled-hef")
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "hw_arch": "hailo8",
        "hef_sha256": _sha(hef),
        "hef_size_bytes": hef.stat().st_size,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_contract_sha256(
            preprocessing
        ),
    }
    _write_json(root / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set",
        "model_id": model_id,
        "benchmark_task": "detection",
    })
    _write_json(root / "output_contracts.json", {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": model_id,
            "backend": "hailo8",
            "variant": "full",
            "task": "detection",
            "contract_status": "recorded",
            "stage": "raw_head",
            "contract_family": "raw_head",
            "recorded_artifact_sha256": _sha(hef),
            "recorded_artifact_size_bytes": hef.stat().st_size,
            "recorded_artifact_path": "hailo/hailo8/full/compiled.hef",
            "endpoint_mode": "raw_detection_head",
            "host_tail_required": True,
            "postprocessing_required": True,
            "requires_external_postprocess": True,
            "output_format": "raw_detection_tensors",
        }],
    })
    onnx = root / f"models/{model_id}.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"source-onnx-used-only-for-byte-binding")
    receipt["source_onnx_sha256"] = _sha(onnx)
    receipt["compiler_onnx_sha256"] = _sha(onnx)
    _write_json(hef.parent / "hailo_hef_build_receipt.json", receipt)
    for index in range(image_count):
        image = root / "resources/validation/images" / f"{index:012d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
    return root


def _matrix(tmp_path: Path) -> Path:
    path = tmp_path / "hardware_matrix.json"
    _write_json(path, {
        "schema": "onnx-splitpoint/evaluation-hardware-matrix",
        "hardware_targets": [{
            "id": "test-hailo8",
            "accelerator": "hailo8",
            "enabled": True,
            "remote": {
                "enabled": True,
                "host": "hailo.example.invalid",
                "user": "runner",
                "port": 2222,
                "remote_base_dir": "~/splitpoint_runs",
                "remote_venv": "source ~/hailo_py/bin/activate",
            },
        }],
    })
    return path


def _config(mod, benchmark_set: Path, out: Path, **kwargs):
    return mod.CanaryConfig(
        benchmark_sets=(benchmark_set,),
        out_dir=out,
        **kwargs,
    )


def _fake_bundle(mod, plans, out_dir):
    bundle = out_dir / "payload.tar.gz"
    bundle.write_bytes(b"payload")
    inventory = [{
        "index": index,
        "model_id": plan.model_id,
        "task": plan.task,
        "benchmark_set_relative": f"benchmark_sets/{index:02d}_{plan.model_id}",
        "images": [{
            "relative_path": f"images/{image.source.name}",
            "source_name": image.source.name,
            "sha256": image.sha256,
        } for image in plan.images],
    } for index, plan in enumerate(plans)]
    manifest = {
        "files": [{
            "path": "placeholder",
            "sha256": hashlib.sha256(b"payload").hexdigest(),
            "size_bytes": 7,
        }],
        "models": inventory,
    }
    return mod.PayloadBundle(
        path=bundle,
        sha256=_sha(bundle),
        size_bytes=bundle.stat().st_size,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
        verifier=VERIFIER,
        verifier_sha256=_sha(VERIFIER),
        verifier_size_bytes=VERIFIER.stat().st_size,
    )


def test_admission_uses_real_hef_receipt_contract_and_caps_at_sixteen(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path, image_count=20)
    before = {
        str(path): _sha(path)
        for path in benchmark_set.rglob("*") if path.is_file()
    }

    plan = mod._attest_benchmark_set(benchmark_set, max_images=16)

    assert plan.model_id == "yolo26m"
    assert plan.task == "detection"
    assert plan.hef_sha256 == _sha(plan.hef)
    assert plan.onnx is not None and plan.onnx_sha256 == _sha(plan.onnx)
    assert plan.output_contract_resolution["contract_resolution_status"] == "attested"
    assert len(plan.images) == 16
    assert {
        str(path): _sha(path)
        for path in benchmark_set.rglob("*") if path.is_file()
    } == before


def test_admission_rejects_hef_symlink(mod, tmp_path: Path) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    hef = benchmark_set / "hailo/hailo8/full/compiled.hef"
    target = tmp_path / "outside.hef"
    target.write_bytes(hef.read_bytes())
    hef.unlink()
    hef.symlink_to(target)

    with pytest.raises(mod.CanaryError, match="unsafe_symlink_hailo8_full_hef"):
        mod._attest_benchmark_set(benchmark_set, max_images=1)


def test_admission_rejects_receipt_hash_drift(mod, tmp_path: Path) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    receipt = (
        benchmark_set
        / "hailo/hailo8/full/hailo_hef_build_receipt.json"
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["hef_sha256"] = "0" * 64
    _write_json(receipt, payload)

    with pytest.raises(
        mod.CanaryError, match="hailo_build_receipt_attestation_failed",
    ):
        mod._attest_benchmark_set(benchmark_set, max_images=1)


def test_admission_rejects_source_onnx_receipt_hash_mismatch(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    receipt = (
        benchmark_set
        / "hailo/hailo8/full/hailo_hef_build_receipt.json"
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["source_onnx_sha256"] = "0" * 64
    _write_json(receipt, payload)

    with pytest.raises(
        mod.CanaryError, match="receipt_source_onnx_binding_failed",
    ):
        mod._attest_benchmark_set(benchmark_set, max_images=1)


def test_admission_rejects_receipt_bound_missing_source_onnx(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    (benchmark_set / "models/yolo26m.onnx").unlink()

    with pytest.raises(
        mod.CanaryError, match="receipt_source_onnx_missing",
    ):
        mod._attest_benchmark_set(benchmark_set, max_images=1)


def test_hardware_matrix_requires_exact_setup_when_ambiguous(
    mod, tmp_path: Path,
) -> None:
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    second = json.loads(json.dumps(payload["hardware_targets"][0]))
    second["id"] = "second-hailo8"
    payload["hardware_targets"].append(second)
    _write_json(matrix, payload)

    with pytest.raises(
        mod.CanaryError, match="multiple_hailo8_remotes_require_setup_id",
    ):
        mod._select_remote_setup(matrix, "")
    selected = mod._select_remote_setup(matrix, "second-hailo8")
    assert selected.setup_id == "second-hailo8"
    assert selected.host == "hailo.example.invalid"


def test_run_canary_pass_is_read_only_and_writes_only_output(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    before = {
        str(path): _sha(path)
        for path in benchmark_set.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(
        mod, "_build_payload",
        lambda plans, out_dir: _fake_bundle(mod, plans, out_dir),
    )

    def execute(plans, out_dir, timeout_s, setup, payload):
        assert timeout_s == 300
        assert setup is None
        assert payload.path.parent == out_dir
        assert len(payload.manifest["models"]) == 1
        return {
            "status": "COMPLETED",
            "runtime_admitted": True,
            "invocations": [{
                "model_id": plans[0].model_id,
                "image": plans[0].images[0].source.name,
                "image_sha256": plans[0].images[0].sha256,
                "returncode": 0,
                "result_relative": "results/00_yolo26m/00",
            }],
        }

    def validate(plans, out_dir, execution, setup_id):
        assert setup_id == "local_hailo8"
        return [{
            "model_id": plans[0].model_id,
            "image": plans[0].images[0].source.name,
            "status": "PASS",
        }]

    result, rc = mod.run_canary(
        _config(mod, benchmark_set, tmp_path / "out-pass"),
        execute=execute,
        validate=validate,
    )

    assert rc == 0
    assert result["status"] == "PASS"
    assert result["artifact_policy"] == "cache_verify_only"
    assert result["compile_or_build_entry_points_allowed"] == []
    assert result["attested_source_files_modified"] is False
    assert result["numerical_correctness_attested"] is False
    assert result["backend_parity_attested"] is False
    assert {
        str(path): _sha(path)
        for path in benchmark_set.rglob("*") if path.is_file()
    } == before
    assert not any(path.parent == benchmark_set for path in (tmp_path / "out-pass").rglob("*"))


@pytest.mark.parametrize(
    ("require_hardware", "expected_status", "expected_rc"),
    [(False, "SKIP", 0), (True, "FAIL", 2)],
)
def test_pre_admission_hardware_skip_policy(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    require_hardware: bool, expected_status: str, expected_rc: int,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    monkeypatch.setattr(
        mod, "_build_payload",
        lambda plans, out_dir: _fake_bundle(mod, plans, out_dir),
    )

    def execute(*_args):
        return {
            "status": "SKIP",
            "reason": "local_hailo8_runtime_unavailable",
            "runtime_admitted": False,
        }

    result, rc = mod.run_canary(
        _config(
            mod, benchmark_set,
            tmp_path / f"out-skip-{int(require_hardware)}",
            require_hardware=require_hardware,
        ),
        execute=execute,
    )

    assert rc == expected_rc
    assert result["status"] == expected_status
    assert result["reason"] == "local_hailo8_runtime_unavailable"


def test_post_admission_execution_error_is_fail(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    monkeypatch.setattr(
        mod, "_build_payload",
        lambda plans, out_dir: _fake_bundle(mod, plans, out_dir),
    )

    def execute(*_args):
        return {
            "status": "FAIL",
            "reason": "remote_smoke_failed:yolo26m:0:rc=1",
            "runtime_admitted": True,
        }

    result, rc = mod.run_canary(
        _config(mod, benchmark_set, tmp_path / "out-fail"),
        execute=execute,
    )

    assert rc == 1
    assert result["status"] == "FAIL"
    assert "remote_smoke_failed" in result["reason"]


def test_output_summary_requires_nonempty_finite_tensors(mod) -> None:
    assert mod._finite_output_summary({
        "head": {"shape": [1, 80, 80, 64], "min": -1.0, "max": 1.0},
    })
    assert not mod._finite_output_summary({})
    assert not mod._finite_output_summary({
        "head": {"shape": [1, 6], "min": float("nan"), "max": 1.0},
    })


@pytest.mark.parametrize(
    "value",
    [
        "source ~/v/bin/activate; touch /tmp/pwn",
        "$(touch /tmp/pwn)",
        "source ~/v/bin/activate\ntrue",
        "source ~/v/bin/activate\x00true",
        "export ONNX_SPLITPOINT_ARTIFACT_POLICY=normal; source ~/v/bin/activate",
    ],
)
def test_remote_venv_rejects_shell_injection_and_policy_override(
    mod, value: str,
) -> None:
    with pytest.raises(mod.CanaryError, match="remote_venv"):
        mod._normalise_remote_venv(value)


def test_remote_venv_normalises_path_and_allowlisted_legacy_snippet(mod) -> None:
    assert mod._normalise_remote_venv("~/venvs/h8/bin/activate") == (
        'source "$HOME"/venvs/h8/bin/activate'
    )
    assert mod._normalise_remote_venv("/venvs/h8/bin/activate") == (
        "source /venvs/h8/bin/activate"
    )
    assert mod._normalise_remote_venv(
        "source ~/hailo_py/bin/activate"
    ) == 'source "$HOME"/hailo_py/bin/activate'
    prefix = mod._remote_policy_activation_prefix(
        mod._normalise_remote_venv("~/venvs/h8/bin/activate")
    )
    assert prefix.index("export ONNX_SPLITPOINT_ARTIFACT_POLICY") < prefix.index(
        "source"
    )
    assert "readonly ONNX_SPLITPOINT_ARTIFACT_POLICY" in prefix


def test_remote_matrix_rejects_ssh_extra_args(mod, tmp_path: Path) -> None:
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    payload["hardware_targets"][0]["remote"]["ssh_extra_args"] = (
        "-o ProxyCommand='touch /tmp/pwn'"
    )
    _write_json(matrix, payload)
    with pytest.raises(mod.CanaryError, match="remote_ssh_extra_args_forbidden"):
        mod._select_remote_setup(matrix, "test-hailo8")


@pytest.mark.parametrize(("field", "value"), (
    ("host", "-unsafe-host"),
    ("user", "-unsafe-user"),
))
def test_remote_matrix_rejects_option_like_host_or_user(
    mod, tmp_path: Path, field: str, value: str,
) -> None:
    matrix = _matrix(tmp_path)
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    payload["hardware_targets"][0]["remote"][field] = value
    _write_json(matrix, payload)
    with pytest.raises(mod.CanaryError, match="remote_host_invalid"):
        mod._select_remote_setup(matrix, "test-hailo8")


def test_setup_id_without_matrix_fails_before_output_creation(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    out = tmp_path / "out"
    result, rc = mod.run_canary(
        _config(mod, benchmark_set, out, setup_id="test-hailo8")
    )
    assert rc == 1
    assert result["status"] == "FAIL"
    assert "setup_id_requires_hardware_matrix" in result["reason"]
    assert not out.exists()


def _payload_manifest(files: dict[str, bytes]) -> bytes:
    payload = {
        "schema": "onnx-splitpoint/v27713-hailo8-structural-canary-payload",
        "schema_version": 1,
        "files": [
            {
                "path": name,
                "kind": "test",
                "model_id": "model",
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for name, raw in sorted(files.items())
        ],
        "models": [],
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


def _write_payload_archive(
    path: Path,
    *,
    files: dict[str, bytes],
    manifest_raw: bytes,
    additions: list[tuple[str, bytes]] | None = None,
    omit: set[str] | None = None,
) -> None:
    members = [
        (f"payload/{name}", raw)
        for name, raw in sorted(files.items())
        if name not in (omit or set())
    ]
    members.append(("payload/canary_payload_manifest.json", manifest_raw))
    members.extend(additions or [])
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in members:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))


@pytest.mark.parametrize("tampered_name", ["model.hef", "image.jpg", "scripts/run.py"])
def test_payload_verifier_rejects_tampered_hef_image_or_script(
    tmp_path: Path, tampered_name: str,
) -> None:
    verifier = _load_verifier()
    declared = {
        "model.hef": b"hef",
        "image.jpg": b"image",
        "scripts/run.py": b"print('safe')\n",
    }
    manifest_raw = _payload_manifest(declared)
    observed = dict(declared)
    observed[tampered_name] += b"-tampered"
    archive = tmp_path / f"tampered-{Path(tampered_name).name}.tar.gz"
    _write_payload_archive(
        archive, files=observed, manifest_raw=manifest_raw,
    )
    with pytest.raises(
        verifier.PayloadVerificationError,
        match="payload_member_(size|sha256)_mismatch",
    ):
        verifier.verify_and_extract(
            archive,
            tmp_path / f"extract-{Path(tampered_name).name}",
            expected_archive_sha256=_sha(archive),
            expected_archive_size=archive.stat().st_size,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )


@pytest.mark.parametrize("case", ["duplicate", "escape", "missing", "extra"])
def test_payload_verifier_rejects_nonexact_or_unsafe_member_sets(
    tmp_path: Path, case: str,
) -> None:
    verifier = _load_verifier()
    files = {"model.hef": b"hef"}
    manifest_raw = _payload_manifest(files)
    additions: list[tuple[str, bytes]] = []
    omit: set[str] = set()
    if case == "duplicate":
        additions.append(("payload/model.hef", b"hef"))
    elif case == "escape":
        additions.append(("payload/../escape", b"x"))
    elif case == "missing":
        omit.add("model.hef")
    elif case == "extra":
        additions.append(("payload/extra", b"x"))
    archive = tmp_path / f"{case}.tar.gz"
    _write_payload_archive(
        archive,
        files=files,
        manifest_raw=manifest_raw,
        additions=additions,
        omit=omit,
    )
    with pytest.raises(verifier.PayloadVerificationError):
        verifier.verify_and_extract(
            archive,
            tmp_path / f"extract-{case}",
            expected_archive_sha256=_sha(archive),
            expected_archive_size=archive.stat().st_size,
            expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        )


def test_payload_verifier_rejects_manifest_duplicate_keys_and_absolute_path(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    duplicate_raw = (
        b'{"schema":"onnx-splitpoint/v27713-hailo8-structural-canary-payload",'
        b'"schema":"duplicate","schema_version":1,"files":[],"models":[]}\n'
    )
    duplicate_archive = tmp_path / "duplicate-json.tar.gz"
    _write_payload_archive(
        duplicate_archive, files={}, manifest_raw=duplicate_raw,
    )
    with pytest.raises(
        verifier.PayloadVerificationError, match="duplicate_json",
    ):
        verifier.verify_and_extract(
            duplicate_archive,
            tmp_path / "extract-duplicate-json",
            expected_archive_sha256=_sha(duplicate_archive),
            expected_archive_size=duplicate_archive.stat().st_size,
            expected_manifest_sha256=hashlib.sha256(duplicate_raw).hexdigest(),
        )

    absolute_manifest = {
        "schema": "onnx-splitpoint/v27713-hailo8-structural-canary-payload",
        "schema_version": 1,
        "files": [{
            "path": "/absolute",
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }],
        "models": [],
    }
    absolute_raw = (json.dumps(absolute_manifest) + "\n").encode()
    absolute_archive = tmp_path / "absolute-manifest-path.tar.gz"
    _write_payload_archive(
        absolute_archive,
        files={},
        manifest_raw=absolute_raw,
        additions=[("payload/absolute", b"x")],
    )
    with pytest.raises(
        verifier.PayloadVerificationError, match="manifest_path_invalid",
    ):
        verifier.verify_and_extract(
            absolute_archive,
            tmp_path / "extract-absolute",
            expected_archive_sha256=_sha(absolute_archive),
            expected_archive_size=absolute_archive.stat().st_size,
            expected_manifest_sha256=hashlib.sha256(absolute_raw).hexdigest(),
        )


def test_real_payload_manifest_binds_hef_onnx_images_and_runtime_scripts(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    plan = mod._attest_benchmark_set(benchmark_set, max_images=1)
    out = tmp_path / "payload-out"
    out.mkdir()
    payload = mod._build_payload((plan,), out)
    rows = {
        row["path"]: row for row in payload.manifest["files"]
    }
    model = payload.manifest["models"][0]
    assert rows[model["hef_path"]]["sha256"] == plan.hef_sha256
    assert rows[model["onnx_path"]]["sha256"] == plan.onnx_sha256
    assert rows[model["images"][0]["payload_path"]]["sha256"] == (
        plan.images[0].sha256
    )
    assert rows["scripts/smoke_hailo10_hef_runner.py"]["sha256"] == _sha(
        ROOT / "scripts/smoke_hailo10_hef_runner.py"
    )


def test_real_multimodel_payload_materializes_local_image_hardlinks(
    mod, tmp_path: Path,
) -> None:
    first_set = _benchmark_set(tmp_path / "first", image_count=1)
    second_set = _benchmark_set(
        tmp_path / "second", image_count=1, model_id="yolo11l",
    )
    first_image = next(
        (first_set / "resources/validation").rglob("*.jpg")
    )
    second_image = next(
        (second_set / "resources/validation").rglob("*.jpg")
    )
    second_image.unlink()
    os.link(first_image, second_image)
    assert first_image.stat().st_ino == second_image.stat().st_ino

    first = mod._attest_benchmark_set(first_set, max_images=1)
    second = mod._attest_benchmark_set(second_set, max_images=1)
    out = tmp_path / "multimodel-payload-out"
    out.mkdir()

    payload = mod._build_payload((first, second), out)
    image_names = [
        f"payload/{model['images'][0]['payload_path']}"
        for model in payload.manifest["models"]
    ]
    with tarfile.open(payload.path, "r:gz") as archive:
        members = [archive.getmember(name) for name in image_names]
        assert all(member.isfile() and not member.islnk() for member in members)
        assert all(not member.linkname for member in members)
        extracted = [archive.extractfile(member) for member in members]
        assert all(handle is not None for handle in extracted)
        assert [handle.read() for handle in extracted if handle is not None] == [
            first_image.read_bytes(), first_image.read_bytes(),
        ]


def test_validation_image_symlink_remains_fail_closed(mod, tmp_path: Path) -> None:
    benchmark_set = _benchmark_set(tmp_path, image_count=1)
    image = next((benchmark_set / "resources/validation").rglob("*.jpg"))
    target = tmp_path / "outside.jpg"
    target.write_bytes(b"outside")
    image.unlink()
    image.symlink_to(target)

    with pytest.raises(mod.CanaryError, match="unsafe_symlink_validation_image"):
        mod._attest_benchmark_set(benchmark_set, max_images=1)


def test_fd_bound_payload_helpers_never_follow_source_symlinks(
    mod, tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"must-not-be-followed")
    link = tmp_path / "source-link.bin"
    link.symlink_to(target)

    with pytest.raises(mod.CanaryError, match="unsafe_payload_source"):
        mod._payload_source_identity(link)
    with tarfile.open(tmp_path / "out.tar.gz", "w:gz") as archive:
        with pytest.raises(mod.CanaryError, match="unsafe_payload_source"):
            mod._add_tar_regular_file(
                archive,
                source=link,
                name="payload/source-link.bin",
                expected_size=target.stat().st_size,
                expected_mode=0o644,
            )


def test_source_drift_after_payload_build_forces_fail_even_from_skip(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    monkeypatch.setattr(
        mod, "_build_payload",
        lambda plans, out_dir: _fake_bundle(mod, plans, out_dir),
    )

    def execute(plans, *_args):
        plans[0].images[0].source.write_bytes(b"mutated-after-snapshot")
        return {
            "status": "SKIP",
            "reason": "local_hailo8_runtime_unavailable",
            "runtime_admitted": False,
        }

    result, rc = mod.run_canary(
        _config(mod, benchmark_set, tmp_path / "out-drift"),
        execute=execute,
    )
    assert rc == 1
    assert result["status"] == "FAIL"
    assert result["attested_source_files_modified"] is True
    assert result["attested_source_files_changed"] == [
        str(benchmark_set / "resources/validation/images/000000000000.jpg")
    ]


def test_remote_completed_result_requires_exact_local_payload_attestation(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    matrix = _matrix(tmp_path)
    monkeypatch.setattr(
        mod, "_build_payload",
        lambda plans, out_dir: _fake_bundle(mod, plans, out_dir),
    )

    def execute(*_args):
        return {
            "status": "COMPLETED",
            "runtime_admitted": True,
            "payload_verified": True,
            "payload_attestation": {
                "status": "PASS",
                "archive_sha256": "0" * 64,
                "archive_size_bytes": 7,
                "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
                "file_count": 1,
            },
            "invocations": [],
        }

    result, rc = mod.run_canary(
        _config(
            mod, benchmark_set, tmp_path / "out-remote-binding",
            hardware_matrix=matrix, setup_id="test-hailo8",
        ),
        execute=execute,
        validate=lambda *_args: pytest.fail("validator must not run"),
    )
    assert rc == 1
    assert result["status"] == "FAIL"
    assert "remote_payload_attestation_not_bound" in result["reason"]


def test_invocation_set_and_result_paths_are_exact_and_unique(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path, image_count=2)
    plan = mod._attest_benchmark_set(benchmark_set, max_images=2)
    duplicate = {
        "model_id": plan.model_id,
        "image": plan.images[0].source.name,
        "image_sha256": plan.images[0].sha256,
        "result_relative": "results/a",
    }
    with pytest.raises(mod.CanaryError, match="invocation_exact_set"):
        mod._validate_results(
            (plan,), tmp_path,
            {
                "runtime_admitted": True,
                "invocations": [duplicate, {**duplicate, "result_relative": "results/b"}],
            },
            "local_hailo8",
        )
    second = {
        "model_id": plan.model_id,
        "image": plan.images[1].source.name,
        "image_sha256": plan.images[1].sha256,
        "result_relative": "results/a",
    }
    with pytest.raises(mod.CanaryError, match="result_paths_not_unique"):
        mod._validate_results(
            (plan,), tmp_path,
            {
                "runtime_admitted": True,
                "invocations": [duplicate, second],
            },
            "local_hailo8",
        )


def test_dump_validation_binds_preprocess_resolution_and_numeric_summary_to_bytes(
    mod, tmp_path: Path,
) -> None:
    import numpy as np

    benchmark_set = _benchmark_set(tmp_path)
    plan = mod._attest_benchmark_set(benchmark_set, max_images=1)
    result_dir = tmp_path / "results/model/00"
    dump = result_dir / "dump"
    dump.mkdir(parents=True)
    tensor = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    tensor_path = dump / "output_00_head.bin"
    tensor.tofile(tensor_path)
    resolution = dict(plan.output_contract_resolution)
    resolution.pop("declaration_source_name")
    resolution["declaration_source"] = str(
        benchmark_set / "output_contracts.json"
    )
    _write_json(dump / "native_full_outputs_manifest.json", {
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "model": plan.model_id,
        "backend": "native_full_hailo8",
        "setup_id": "local_hailo8",
        "input_image_sha256": plan.images[0].sha256,
        "authoritative_output_contract_resolution": resolution,
        "outputs": [{
            "name": "head",
            "file": tensor_path.name,
            "dtype": "float32",
            "shape": [1, 3],
            "bytes": tensor.nbytes,
            "sha256": _sha(tensor_path),
        }],
    })
    input_payload = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "model": plan.model_id,
        "task": plan.task,
        "setup_id": "local_hailo8",
        "input_image_sha256": plan.images[0].sha256,
        "preprocessing_contract": plan.preprocessing_contract,
        "preprocessing_contract_sha256": plan.preprocessing_contract_sha256,
    }
    _write_json(dump / "native_full_input_manifest.json", input_payload)
    summary = {
        "head": {
            "shape": [1, 3], "dtype": "float32",
            "min": 1.0, "max": 3.0,
        }
    }
    assert mod._validate_dump_artifacts(
        result_dir, plan=plan, setup_id="local_hailo8",
        image_sha256=plan.images[0].sha256,
        output_summary=summary,
    )
    assert not mod._validate_dump_artifacts(
        result_dir, plan=plan, setup_id="local_hailo8",
        image_sha256=plan.images[0].sha256,
        output_summary={"head": {**summary["head"], "max": 4.0}},
    )
    input_payload["preprocessing_contract_sha256"] = "0" * 64
    _write_json(dump / "native_full_input_manifest.json", input_payload)
    assert not mod._validate_dump_artifacts(
        result_dir, plan=plan, setup_id="local_hailo8",
        image_sha256=plan.images[0].sha256,
        output_summary=summary,
    )


def test_remote_executor_verifies_payload_then_bounds_inference_before_cleanup(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    plan = mod._attest_benchmark_set(benchmark_set, max_images=1)
    out = tmp_path / "remote-out"
    out.mkdir()
    payload = _fake_bundle(mod, (plan,), out)
    model = payload.manifest["models"][0]
    model.update({
        "hef_path": "benchmark_sets/00_yolo26m/hailo/hailo8/full/compiled.hef",
        "hef_sha256": plan.hef_sha256,
    })
    expected = mod._expected_payload_attestation(payload)
    commands: list[tuple[str, int]] = []

    class Registry:
        def active_count(self):
            return 0

    class Transport:
        def test_connection(self, timeout_s):
            return True, "ok"

        def run_read_only(self, command, timeout_s):
            return 0, ""

        def resolve_path_read_only(self, value, timeout_s):
            return "/safe/canary/root"

        def scp_upload(self, source, target, timeout_s):
            return 0, ""

        def scp_download(self, source, target, timeout_s):
            Path(target).write_bytes(b"archive")
            return 0, ""

        def run(self, command, timeout_s):
            commands.append((command, timeout_s))
            if "verify_payload.py --archive" in command:
                return 0, json.dumps(expected)
            if "assert 0 < s.st_size" in command:
                return 0, json.dumps({
                    "sha256": hashlib.sha256(b"archive").hexdigest(),
                    "size_bytes": len(b"archive"),
                })
            return 0, ""

    monkeypatch.setattr(
        mod, "_remote_transport",
        lambda setup, session_id: (Transport(), Registry()),
    )
    monkeypatch.setattr(mod, "_safe_extract", lambda *_args: None)
    setup = mod.RemoteSetup(
        setup_id="test-hailo8",
        host="hailo.invalid",
        user="runner",
        port=22,
        remote_base_dir="/safe/canary/root",
        remote_venv=mod._normalise_remote_venv("~/venvs/h8/bin/activate"),
        ssh_extra_args="",
    )
    execution = mod._execute_remote(
        (plan,), out, 300, setup, payload,
    )
    assert execution["status"] == "COMPLETED"
    assert execution["runtime_admitted"] is True
    assert execution["payload_verified"] is True
    inference = [
        row for row in commands
        if "smoke_hailo10_full_from_benchmarkset.py" in row[0]
    ]
    assert len(inference) == 1
    command, outer_timeout = inference[0]
    assert "timeout --signal=TERM --kill-after=15s 300s" in command
    assert outer_timeout == 330
    assert command.index("export ONNX_SPLITPOINT_ARTIFACT_POLICY") < command.index(
        "source"
    )
    assert commands[-1][0].startswith("rm -rf -- ")


def test_safe_result_extract_rejects_duplicate_and_traversal(
    mod, tmp_path: Path,
) -> None:
    for case, members in {
        "duplicate": [("results/a", b"1"), ("results/a", b"2")],
        "traversal": [("results/../escape", b"1")],
    }.items():
        archive = tmp_path / f"result-{case}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for name, raw in members:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                handle.addfile(info, io.BytesIO(raw))
        with pytest.raises(mod.CanaryError, match="unsafe_result_archive_member"):
            mod._safe_extract(archive, tmp_path / f"extract-result-{case}")


def test_safe_result_extract_rejects_normalised_path_aliases(
    mod, tmp_path: Path,
) -> None:
    archive = tmp_path / "result-normalised-alias.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        directory = tarfile.TarInfo("results/a/")
        directory.type = tarfile.DIRTYPE
        handle.addfile(directory)
        file_info = tarfile.TarInfo("results/a")
        file_info.size = 1
        handle.addfile(file_info, io.BytesIO(b"x"))

    with pytest.raises(mod.CanaryError, match="unsafe_result_archive_member"):
        mod._safe_extract(archive, tmp_path / "extract-result-normalised-alias")


def test_safe_result_extract_accepts_real_tar_directory_members(
    mod, tmp_path: Path,
) -> None:
    source = tmp_path / "remote-pack/results/model/00"
    source.mkdir(parents=True)
    (source / "report.json").write_text('{"ok":true}\n', encoding="utf-8")
    (source / "tensor.bin").write_bytes(b"tensor")
    archive = tmp_path / "real-results-pack.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(tmp_path / "remote-pack/results", arcname="results")
    member_kinds = [(row.name, row.isdir(), row.isfile()) for row in tarfile.open(archive).getmembers()]
    assert any(is_dir for _name, is_dir, _is_file in member_kinds)
    destination = tmp_path / "download"
    mod._safe_extract(archive, destination)
    assert (destination / "results/model/00/report.json").read_text(
        encoding="utf-8"
    ) == '{"ok":true}\n'
    assert (destination / "results/model/00/tensor.bin").read_bytes() == b"tensor"


def test_output_path_with_symlink_parent_is_rejected(mod, tmp_path: Path) -> None:
    benchmark_set = _benchmark_set(tmp_path)
    real = tmp_path / "real-output-parent/existing"
    real.mkdir(parents=True)
    link = tmp_path / "output-link"
    link.symlink_to(real.parent, target_is_directory=True)
    with pytest.raises(mod.CanaryError, match="unsafe_output_parent"):
        mod._prepare_output(
            link / "existing/new-canary", inputs=(benchmark_set,),
        )
    assert not (real / "new-canary").exists()


def test_benchmark_set_symlink_is_rejected_before_inside_output_mutation(
    mod, tmp_path: Path,
) -> None:
    benchmark_set = _benchmark_set(tmp_path / "source")
    alias = tmp_path / "benchmark-alias"
    alias.symlink_to(benchmark_set, target_is_directory=True)
    outside_target = benchmark_set / "must-not-be-created"
    result, rc = mod.run_canary(
        mod.CanaryConfig(
            benchmark_sets=(alias,),
            out_dir=alias / "must-not-be-created",
        )
    )
    assert rc == 1
    assert result["status"] == "FAIL"
    assert "unsafe_benchmark_set" in result["reason"]
    assert not outside_target.exists()


def test_main_never_prints_pass_when_result_receipt_write_fails(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = mod.CanaryConfig(
        benchmark_sets=(tmp_path / "benchmark",),
        out_dir=tmp_path / "out",
    )
    monkeypatch.setattr(mod, "_parse_args", lambda _argv: config)
    monkeypatch.setattr(
        mod, "run_canary",
        lambda _config: ({
            "schema": mod.SCHEMA,
            "schema_version": mod.SCHEMA_VERSION,
            "status": "PASS",
            "reason": "",
        }, 0),
    )
    monkeypatch.setattr(
        mod, "_write_result",
        lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
    )
    rc = mod.main([])
    output = capsys.readouterr().out
    assert rc == 1
    assert '"status": "FAIL"' in output
    assert "canary_result_receipt_write_failed:OSError:disk full" in output


def test_main_never_claims_receipt_written_when_parent_is_unavailable(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "missing-parent" / "out"
    config = mod.CanaryConfig(
        benchmark_sets=(tmp_path / "benchmark",),
        out_dir=out,
    )
    monkeypatch.setattr(mod, "_parse_args", lambda _argv: config)
    monkeypatch.setattr(
        mod, "run_canary",
        lambda _config: ({
            "schema": mod.SCHEMA,
            "schema_version": mod.SCHEMA_VERSION,
            "status": "SKIP",
            "reason": "optional_material_unavailable",
        }, 0),
    )

    rc = mod.main([])

    output = capsys.readouterr().out
    assert rc == 0
    assert '"status": "SKIP"' in output
    assert '"result_receipt_written": false' in output
    assert not (out / "canary_result.json").exists()
    assert "V27713_HAILO8_ARTIFACT_CANARY=SKIP" in output


def test_main_keeps_full_source_list_in_receipt_but_compacts_stdout(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    checked = ["/attested/source/a", "/attested/source/b"]
    config = mod.CanaryConfig(
        benchmark_sets=(tmp_path / "benchmark",),
        out_dir=out,
    )
    monkeypatch.setattr(mod, "_parse_args", lambda _argv: config)
    monkeypatch.setattr(
        mod, "run_canary",
        lambda _config: ({
            "schema": mod.SCHEMA,
            "schema_version": mod.SCHEMA_VERSION,
            "status": "SKIP",
            "reason": "optional_material_unavailable",
            "attested_source_file_count": len(checked),
            "attested_source_files_checked": checked,
        }, 0),
    )

    assert mod.main([]) == 0

    output = capsys.readouterr().out
    receipt = json.loads(
        (out / "canary_result.json").read_text(encoding="utf-8")
    )
    assert receipt["attested_source_files_checked"] == checked
    assert all(path not in output for path in checked)
    assert '"attested_source_files_checked_omitted_from_stdout": 2' in output
    assert '"attested_source_file_count": 2' in output
