from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import shlex
import subprocess
from typing import Any

import onnx_splitpoint_tool.resume_artifact_rehydration as rehydration
import onnx_splitpoint_tool.resume_remote_rehydration as remote
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
    build_resume_artifact_stage_map,
    requirements_from_contract_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "v2713_resume_failure"
RESUME_RUNNER = ROOT / "scripts" / "run_native_producer_energy_from_summary.py"
REMOTE_RUN_ROOT = (
    "/home/nx/native_fifo_evalsets/"
    "resnet_yolo26s_yolo7_20260727_153940"
)

SPLIT_FILE_SHA256 = (
    "61ee251ff05550a25dce94f35f6ed0a49c0343a9e0d196d9b5dd72a7143b7dd2"
)
SPLIT_CONTRACT_SHA256 = (
    "97b539cafb007cd1bd03944df1bd17647f51caa7678fe309a1dd1bbb30a46a1c"
)
MANIFEST_SHA256 = (
    "d3594580372142ea9cbaaa144642a41d0f7f2b74c4a51d46662b80ae4adf3b8c"
)
PAYLOAD_SHA256 = (
    "66646c62bbdbab5db3811d5fd46cb791d0cc01d1496908cda5e7cc871d14b485"
)
FULL_FILE_SHA256 = (
    "2bbc21804c7558f4d84558465095cf096297c678b23f37bfd3a41f2a4d819033"
)
FULL_CONTRACT_SHA256 = (
    "ed4e01dc5ebab5332e3e6808e82aff4dd80f55d4343662f63d5d5ef298de1a7f"
)
HEF_SHA256 = (
    "edb441cbce75ac1aaa3ce489142751927ceeea4af79a5b98f9a1b4501de35abd"
)

SPLIT_MANIFEST_REMOTE = (
    f"{REMOTE_RUN_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
    "hailo10h_to_trt/float32_layout_fp16/native_outputs/"
    "native_outputs_manifest.json"
)
SPLIT_PAYLOAD_REMOTE = (
    f"{REMOTE_RUN_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
    "hailo10h_to_trt/float32_layout_fp16/native_outputs/"
    "output_00_output0.bin"
)
FULL_HEF_REMOTE = (
    f"{REMOTE_RUN_ROOT}/yolov7_paper/benchmark_set/hailo/hailo10/"
    "full/compiled.hef"
)

# Exact execution-context options from the old 2.71.2 plan.cmd.  The old plan
# predates the explicit execution_context payload, so this command is the
# authoritative backwards-compatible source.
OLD_PLAN_CMD = [
    "/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python",
    "-u",
    "/home/kmika/ONNX-Splitpoint-Tool/scripts/native_producer_energy_plan.py",
    "--hailo8-ssh",
    "nx@192.168.0.104",
    "--hailo10-ssh",
    "nx@192.168.0.145",
    "--deepx-ssh",
    "nx@192.168.0.102",
    "--hailo8-env",
    "source ~/hailo_py/bin/activate",
    "--hailo10-env",
    "source ~/venvs/hailo10/bin/activate",
    "--deepx-env",
    "source ~/venvs/deepx-runtime/bin/activate",
    "--engine-build-python",
    "auto",
    "--remote-tool-dir",
    "/home/nx/ONNX-Splitpoint-Tool",
    "--remote-root",
    REMOTE_RUN_ROOT,
]


def _fixture_bytes(name: str) -> bytes:
    encoded = "".join(
        (FIXTURES / name).read_text(encoding="ascii").split()
    )
    return gzip.decompress(base64.b64decode(encoded, validate=True))


def _fixture_json(name: str) -> dict[str, Any]:
    value = json.loads(_fixture_bytes(name).decode("utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mirror_write(root: Path, remote_path: str, value: bytes) -> Path:
    target = root.joinpath(*PurePosixPath(remote_path).parts[1:])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    return target


class _LocalSshRunner:
    def __init__(self) -> None:
        self.markers: list[str] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        remote_argv = shlex.split(command[-1])
        marker = next(
            (
                line.strip()
                for line in remote_argv[2].splitlines()
                if "ONNX_SPLITPOINT_RESUME_REMOTE_" in line
            ),
            "",
        )
        self.markers.append(marker)
        return subprocess.run(
            remote_argv,
            stdin=kwargs.get("stdin"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=kwargs.get("timeout"),
        )


def test_exact_uploaded_failure_pack_fixture_bytes_are_sealed() -> None:
    expected = {
        "split_command_contract.json.gz.b64": (141016, SPLIT_FILE_SHA256),
        "full_command_contract.json.gz.b64": (11062, FULL_FILE_SHA256),
        "yolo26_b038_native_outputs_manifest.json.gz.b64": (
            6378,
            MANIFEST_SHA256,
        ),
        "yolo26_b038_output_00_output0.bin.gz.b64": (
            7200,
            PAYLOAD_SHA256,
        ),
    }
    for name, (size, digest) in expected.items():
        value = _fixture_bytes(name)
        assert len(value) == size
        assert _sha256(value) == digest


def test_old_plan_command_binds_exact_root_and_selector_identities() -> None:
    runner = runpy.run_path(str(RESUME_RUNNER))
    context = runner["_canonical_resume_execution_context"]({
        "plan_payload": {},
        "plan": {"cmd": list(OLD_PLAN_CMD)},
    })
    assert context["remote_root"] == REMOTE_RUN_ROOT
    assert context["remote_tool_dir"] == "/home/nx/ONNX-Splitpoint-Tool"
    assert context["hailo10_ssh"] == "nx@192.168.0.145"
    assert context["hailo10_env"] == "source ~/venvs/hailo10/bin/activate"

    split = _fixture_json("split_command_contract.json.gz.b64")
    full = _fixture_json("full_command_contract.json.gz.b64")
    identities = [
        tuple(row[field] for field in ("backend", "model", "case", "setup_id"))
        for row in (split, full)
    ]
    assert identities == [
        (
            "hailo10h_to_trt",
            "yolo26s",
            "b038",
            "orin_nx_hailo10_01",
        ),
        (
            "native_full_hailo10h",
            "yolov7_paper",
            "full",
            "orin_nx_hailo10_01",
        ),
    ]
    assert [
        runner["_resume_selector_text"](identity) for identity in identities
    ] == [
        "hailo10h_to_trt|yolo26s|b038|orin_nx_hailo10_01",
        "native_full_hailo10h|yolov7_paper|full|orin_nx_hailo10_01",
    ]
    assert split["contract_sha256"] == SPLIT_CONTRACT_SHA256
    assert full["contract_sha256"] == FULL_CONTRACT_SHA256


def test_split_contract_resolves_exact_manifest_and_payload(
    tmp_path: Path,
) -> None:
    split = _fixture_json("split_command_contract.json.gz.b64")
    manifest = _fixture_bytes(
        "yolo26_b038_native_outputs_manifest.json.gz.b64"
    )
    payload = _fixture_bytes("yolo26_b038_output_00_output0.bin.gz.b64")
    requirements = requirements_from_contract_artifacts(
        split,
        roles=("semantic_output_manifest",),
    )
    assert requirements == [
        ArtifactRequirement(
            role="semantic_output_manifest",
            remote_path=SPLIT_MANIFEST_REMOTE,
            sha256=MANIFEST_SHA256,
            size_bytes=6378,
        )
    ]

    mirror = tmp_path / "sealed-old-run-mirror"
    _mirror_write(mirror, SPLIT_MANIFEST_REMOTE, manifest)
    _mirror_write(mirror, SPLIT_PAYLOAD_REMOTE, payload)
    stage_map = build_resume_artifact_stage_map(
        requirements,
        run_mirror_roots=(mirror,),
        artifact_store_roots=(),
        allowed_remote_roots=(REMOTE_RUN_ROOT,),
    )

    assert stage_map["status"] == "ready"
    assert stage_map["artifact_count"] == 2
    assert stage_map["total_bytes"] == 6378 + 7200
    entries = {
        row["remote_path"]: row for row in stage_map["entries"]
    }
    assert set(entries) == {SPLIT_MANIFEST_REMOTE, SPLIT_PAYLOAD_REMOTE}
    assert entries[SPLIT_MANIFEST_REMOTE]["sha256"] == MANIFEST_SHA256
    assert entries[SPLIT_MANIFEST_REMOTE]["size_bytes"] == 6378
    assert entries[SPLIT_PAYLOAD_REMOTE]["sha256"] == PAYLOAD_SHA256
    assert entries[SPLIT_PAYLOAD_REMOTE]["size_bytes"] == 7200
    assert entries[SPLIT_PAYLOAD_REMOTE]["roles"] == [
        "semantic_output_manifest.payload[0]:outputs[0]"
    ]


def test_real_split_fixture_rehydrates_after_remote_run_root_cleanup(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    run_root = storage_root / "resnet_yolo26s_yolo7_20260727_153940"
    mapped_manifest = SPLIT_MANIFEST_REMOTE.replace(
        REMOTE_RUN_ROOT, str(run_root), 1,
    )
    mapped_payload = SPLIT_PAYLOAD_REMOTE.replace(
        REMOTE_RUN_ROOT, str(run_root), 1,
    )
    manifest = _fixture_bytes(
        "yolo26_b038_native_outputs_manifest.json.gz.b64"
    )
    payload = _fixture_bytes("yolo26_b038_output_00_output0.bin.gz.b64")
    mirror = tmp_path / "sealed-old-run-mirror"
    _mirror_write(mirror, mapped_manifest, manifest)
    _mirror_write(mirror, mapped_payload, payload)
    stage_map = build_resume_artifact_stage_map(
        [
            ArtifactRequirement(
                role="semantic_output_manifest",
                remote_path=mapped_manifest,
                sha256=MANIFEST_SHA256,
                size_bytes=len(manifest),
            ),
            ArtifactRequirement(
                role="semantic_output_manifest.payload[0]:outputs[0]",
                remote_path=mapped_payload,
                sha256=PAYLOAD_SHA256,
                size_bytes=len(payload),
            ),
        ],
        run_mirror_roots=(mirror,),
        artifact_store_roots=(),
        allowed_remote_roots=(str(run_root),),
        expand_payload_manifests=False,
    )
    runner = _LocalSshRunner()

    report = remote.rehydrate_remote_stage_map(
        stage_map,
        ssh_target="nx@192.168.0.145",
        remote_run_root=str(run_root),
        resume_attempt_id="resume-real-cleanup-fixture",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["remote_run_root_created"] is True
    assert report["rehydrated_count"] == 2
    assert Path(mapped_manifest).read_bytes() == manifest
    assert Path(mapped_payload).read_bytes() == payload
    assert runner.markers.count(
        "# ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1"
    ) == 1
    assert not any("unsafe_root" in marker for marker in runner.markers)


def test_full_contract_binds_exact_hef_and_legacy_cas_identity() -> None:
    full = _fixture_json("full_command_contract.json.gz.b64")
    requirements = requirements_from_contract_artifacts(
        full,
        roles=("hef",),
    )
    assert requirements == [
        ArtifactRequirement(
            role="hef",
            remote_path=FULL_HEF_REMOTE,
            sha256=HEF_SHA256,
            size_bytes=None,
        )
    ]

    store_root = Path("artifact-store")
    cas_path = (
        rehydration._store_object_directories(store_root, HEF_SHA256)[0]
        / "compiled.hef"
    )
    assert cas_path == (
        store_root
        / "objects"
        / "sha256"
        / "ed"
        / HEF_SHA256
        / "compiled.hef"
    )
