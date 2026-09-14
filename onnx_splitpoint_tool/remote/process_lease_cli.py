"""Warning-free module entry point for the remote lease broker."""

from __future__ import annotations

from .process_lease import process_lease_cli_main


if __name__ == "__main__":
    raise SystemExit(process_lease_cli_main())
