from .base import Backend, PreparedHandle

__all__ = ["Backend", "PreparedHandle", "HailoBackend"]


def __getattr__(name: str):
    """Avoid importing the optional Hailo stack for unrelated backends."""
    if name == "HailoBackend":
        from .hailo_backend import HailoBackend

        return HailoBackend
    raise AttributeError(name)
