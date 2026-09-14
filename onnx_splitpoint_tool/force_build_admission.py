"""Per-start Force consent; deliberately separate from artifact contracts."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .config_values import parse_config_bool, validate_profile_config_booleans


class ForceBuildConsentRequired(ValueError):
    error_code = "force_build_confirmation_required"

    def __init__(self, backends: Iterable[str], *, resume: bool = False) -> None:
        self.backends = tuple(backends)
        self.resume = resume
        names = ", ".join("Hailo" if name == "hailo" else "DeepX" for name in self.backends)
        action = "Resume mit gebundenem Profilsnapshot" if resume else "Neuer Start"
        flags = " ".join(f"--confirm-force-build {name}" for name in self.backends)
        super().__init__(
            f"{self.error_code}: {action}: {names} Force ist AN; kompatible "
            "Cachetreffer werden bewusst übergangen. Für diesen Start erneut "
            f"bestätigen (CLI: {flags}) oder Force in der maßgeblichen "
            "Konfiguration ausschalten."
        )


def force_build_backends(
    profile: Mapping[str, Any], *, hailo_force_build: bool = False,
) -> tuple[str, ...]:
    """Return exact configured backend families; never coerce YAML strings."""
    validate_profile_config_booleans(profile)
    cli_hailo = parse_config_bool(hailo_force_build, field="options.hailo_force_build")
    active = []
    for backend in ("hailo", "deepx"):
        cfg = profile.get(f"{backend}_build", {})
        force = parse_config_bool(cfg.get("force_build", False), field=f"{backend}_build.force_build")
        if force or (backend == "hailo" and cli_hailo):
            active.append(backend)
    return tuple(active)


def require_force_build_consent(
    profile: Mapping[str, Any], *, confirmed_backends: Iterable[str] = (),
    source: str = "api", resume: bool = False, hailo_force_build: bool = False,
) -> dict[str, Any]:
    """Check caller-supplied consent, ignoring anything archived in the profile.

    A confirmation never changes a Force setting or grants another backend.
    The return value belongs to the current execution session, not profile YAML
    or a cache key. Callers must obtain confirmations afresh on every start.
    """
    requested = force_build_backends(profile, hailo_force_build=hailo_force_build)
    if isinstance(confirmed_backends, (str, bytes)):
        raise ValueError("force_build_confirmation_invalid:confirmed_backends")
    confirmed = tuple(confirmed_backends)
    if any(type(item) is not str or item not in {"hailo", "deepx"} for item in confirmed):
        raise ValueError("force_build_confirmation_invalid:confirmed_backends")
    missing = tuple(backend for backend in requested if backend not in confirmed)
    if missing:
        raise ForceBuildConsentRequired(missing, resume=resume)
    return {
        "required_backends": list(requested),
        "confirmed_backends": [backend for backend in requested if backend in confirmed],
        "confirmation_source": str(source) if requested else "not_required",
        "resume_requested": bool(resume),
        "compatible_cache_hits_intentionally_bypassed": bool(requested),
    }
