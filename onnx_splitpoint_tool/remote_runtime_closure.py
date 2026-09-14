"""Single source of truth for remotely staged Native Python modules.

The GUI workflow and the standalone updater execute the same Native runners.
Keeping their package closure in one place prevents one production path from
silently omitting an import that the other path happens to stage.

The inventory is the exact transitive Python closure used by remotely started
Native scripts.  Any newly imported runtime module must be added here before a
release can pass its remote import preflight.
"""

from __future__ import annotations

from typing import Tuple


RemotePackageAsset = Tuple[str, str, Tuple[str, ...]]


NATIVE_REMOTE_PACKAGE_CLOSURE: Tuple[RemotePackageAsset, ...] = (
    ("onnx_splitpoint_tool/deepx/__init__.py", "onnx_splitpoint_tool.deepx", ("DeepX DX-M1",)),
    ("onnx_splitpoint_tool/deepx/config.py", "onnx_splitpoint_tool.deepx.config", ("def classification_profile_admission",)),
    ("onnx_splitpoint_tool/native_job_identity.py", "onnx_splitpoint_tool.native_job_identity", ("def planned_native_identity", "def failed_native_result")),
    (
        "onnx_splitpoint_tool/quality_result_contract.py",
        "onnx_splitpoint_tool.quality_result_contract",
        ("UNCERTAINTY_FIELDS", "def project_quality_result", "def project_flat_quality_uncertainty"),
    ),
    (
        "onnx_splitpoint_tool/validation/host_postprocess.py",
        "onnx_splitpoint_tool.validation.host_postprocess",
        (
            "def resolve_host_postprocess_evidence",
            "def apply_host_postprocess_aliases",
        ),
    ),
    (
        "onnx_splitpoint_tool/validation/accuracy_gates.py",
        "onnx_splitpoint_tool.validation.accuracy_gates",
        (
            "class AccuracyGatePolicy",
            "def evaluate_detection_similarity",
            "def apply_accuracy_gate_to_row",
        ),
    ),
    (
        "onnx_splitpoint_tool/preprocessing_contract.py",
        "onnx_splitpoint_tool.preprocessing_contract",
        (
            "def canonical_image_preprocessing_contract",
            "def prepare_rgb_uint8_image",
            "def runtime_numeric_input_identity",
        ),
    ),
    (
        "onnx_splitpoint_tool/native_split_quality.py",
        "onnx_splitpoint_tool.native_split_quality",
        (
            "validate_native_split_quality_binding",
            "bind_quality_to_native_split",
            "native-split-quality-consumer-attestation",
        ),
    ),
    (
        "onnx_splitpoint_tool/native_command_contract.py",
        "onnx_splitpoint_tool.native_command_contract",
        (
            "verify_native_command_contract",
            "verify_native_energy_command_contract",
            "verify_native_split_part2_input_contract",
            "successful_runtime_argv",
            "split_energy_runtime_argv",
            "load_split_energy_workload_binding",
        ),
    ),
    (
        "onnx_splitpoint_tool/native_output_endpoint.py",
        "onnx_splitpoint_tool.native_output_endpoint",
        (
            "def attest_decoded_nms",
            "def runtime_output_contract",
            "def load_manifest_outputs",
        ),
    ),
    (
        "onnx_splitpoint_tool/runners/_types.py",
        "onnx_splitpoint_tool.runners._types",
        ("class RunCfg", "class BackendRunOut"),
    ),
    (
        "onnx_splitpoint_tool/resources_utils.py",
        "onnx_splitpoint_tool.resources_utils",
        (
            "def persistent_resource_path",
            "Python 3.8 has no importlib.resources.files",
        ),
    ),
    (
        "onnx_splitpoint_tool/runners/backends/base.py",
        "onnx_splitpoint_tool.runners.backends.base",
        ("class PreparedHandle",),
    ),
    (
        "onnx_splitpoint_tool/runners/backends/hailo_utils.py",
        "onnx_splitpoint_tool.runners.backends.hailo_utils",
        ("def get_dfc_manager",),
    ),
    (
        "onnx_splitpoint_tool/config_values.py",
        "onnx_splitpoint_tool.config_values",
        (
            "def parse_config_bool",
            "def validate_profile_config_booleans",
            "config_boolean_invalid:",
        ),
    ),
    (
        "onnx_splitpoint_tool/cache_verify_policy.py",
        "onnx_splitpoint_tool.cache_verify_policy",
        (
            "CACHE_VERIFY_ONLY",
            "cache_miss_blocked_message",
            "compiler_dispatch_forbidden",
        ),
    ),
    (
        "onnx_splitpoint_tool/hailo_attempt_receipts.py",
        "onnx_splitpoint_tool.hailo_attempt_receipts",
        (
            "def finalize_hailo_attempt",
            "class HailoAttempt",
            "def start_hailo_attempt",
        ),
    ),
    (
        "onnx_splitpoint_tool/hailo_timeout_policy.py",
        "onnx_splitpoint_tool.hailo_timeout_policy",
        (
            "HAILO_UNLIMITED_TIMEOUT_TOKENS",
            "def parse_hailo_timeout_seconds",
        ),
    ),
    (
        "onnx_splitpoint_tool/runners/backends/hailo_backend.py",
        "onnx_splitpoint_tool.runners.backends.hailo_backend",
        ("_owned_writable_c_buffer", "class _HailoSession"),
    ),
    (
        "onnx_splitpoint_tool/runners/harness/base.py",
        "onnx_splitpoint_tool.runners.harness.base",
        ("postprocess_result_to_dict",),
    ),
    (
        "onnx_splitpoint_tool/runners/harness/yolo.py",
        "onnx_splitpoint_tool.runners.harness.yolo",
        ("class YoloHarness", "original_wh"),
    ),
    (
        "onnx_splitpoint_tool/runners/native_full_input.py",
        "onnx_splitpoint_tool.runners.native_full_input",
        (
            "prepare_and_seal_deepx_native_full_input",
            "load_sealed_deepx_native_full_input",
            "shared_pre_timing_deepx_input_sealer",
        ),
    ),
    (
        "onnx_splitpoint_tool/runners/native_split_quality_runtime.py",
        "onnx_splitpoint_tool.runners.native_split_quality_runtime",
        (
            "prepare_native_split_quality_binding",
            "cache_miss_blocked:artifact_policy=cache_verify_only",
            "def trt_build_forbidden",
            "compiler_dispatched",
        ),
    ),
    (
        "onnx_splitpoint_tool/native_detection_postprocess.py",
        "onnx_splitpoint_tool.native_detection_postprocess",
        ("FrozenDetectionPostprocessor", "invariant_contract_sha256"),
    ),
    (
        "onnx_splitpoint_tool/native_three_stage.py",
        "onnx_splitpoint_tool.native_three_stage",
        ("class NativeThreeStageError", "THREE_STAGE_RESULT_SCHEMA"),
    ),
    (
        "onnx_splitpoint_tool/hailo_full_contract_promotion.py",
        "onnx_splitpoint_tool.hailo_full_contract_promotion",
        (
            "def promote_verified_hailo_full_contracts",
            "def remote_import_preflight",
            "source_onnx_multiscale_raw_head",
        ),
    ),
    # A cold target must import the same package initializers as the controller.
    # In particular validation.__init__ imports accuracy_gates, and the harness
    # initializer imports classification. Do not rely on a preinstalled tree or
    # replace these initializers with empty namespace markers.
    ("onnx_splitpoint_tool/release_identity.py", "onnx_splitpoint_tool.release_identity", ("BUILD_ID", "VERSION")),
    ("onnx_splitpoint_tool/__init__.py", "onnx_splitpoint_tool", ("__build_id__", "__version__")),
    ("onnx_splitpoint_tool/validation/__init__.py", "onnx_splitpoint_tool.validation", ("from .accuracy_gates import",)),
    ("onnx_splitpoint_tool/runners/__init__.py", "onnx_splitpoint_tool.runners", ("from ._types import",)),
    ("onnx_splitpoint_tool/runners/backends/__init__.py", "onnx_splitpoint_tool.runners.backends", ("from .base import Backend",)),
    ("onnx_splitpoint_tool/runners/harness/classification.py", "onnx_splitpoint_tool.runners.harness.classification", ("class ClassificationHarness",)),
    ("onnx_splitpoint_tool/runners/harness/__init__.py", "onnx_splitpoint_tool.runners.harness", ("from .classification import ClassificationHarness",)),
)


def native_remote_package_closure() -> Tuple[RemotePackageAsset, ...]:
    """Return the immutable, ordered Native remote package inventory."""

    return NATIVE_REMOTE_PACKAGE_CLOSURE
