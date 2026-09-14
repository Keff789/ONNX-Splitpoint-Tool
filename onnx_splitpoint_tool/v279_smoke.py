"""Compatibility entry point for the current release smoke (historical alias)."""
from .v282_smoke import (  # noqa: F401
    BUILD_ID, LINEAGE, NEW_FEATURES, REQUIRED_FEATURES, VERSION, main,
)

if __name__ == "__main__":
    raise SystemExit(main())
