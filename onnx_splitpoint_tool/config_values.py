"""Small, shared type boundary for persisted configuration switches.

Normal profile/registry/CLI loads are strict. Legacy text conversion is an
explicit opt-in for migration tools only, never implicit start consent.
"""
from __future__ import annotations

from typing import Any, Mapping


def parse_config_bool(value: Any, *, field: str, allow_legacy_text: bool = False) -> bool:
    if type(value) is bool:
        return value
    if allow_legacy_text and type(value) is str:
        token = value.strip().lower()
        if token in {"true", "false"}:
            return token == "true"
    raise ValueError(f"config_boolean_invalid:{field}")


def validate_profile_config_booleans(profile: Mapping[str, Any]) -> None:
    """Reject malformed force flags before an override can conceal them."""
    if not isinstance(profile, Mapping):
        raise ValueError("config_mapping_invalid:evaluation_profile")
    for block_name, key in (
        ("hailo_build", "force_build"),
        ("deepx_build", "force_build"),
        ("execution_preset", "follow_tool_config"),
    ):
        if block_name not in profile:
            continue
        block = profile[block_name]
        if not isinstance(block, Mapping):
            raise ValueError(f"config_mapping_invalid:{block_name}")
        if key in block:
            parse_config_bool(block[key], field=f"{block_name}.{key}")
