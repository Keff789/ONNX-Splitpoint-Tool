#!/usr/bin/env python3
"""Publish canonical v2.82 controller/supervisor source into a release bundle."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

FILES = ('install_and_accept_v282.py', 'acceptance_process.py')


def sync(bundle: Path, *, verify_only: bool = False) -> None:
    source = Path(__file__).resolve().parent / 'release_bundle'
    if not bundle.is_dir():
        raise ValueError('release_bundle_directory_missing')
    for name in FILES:
        destination = bundle / name
        if destination.is_symlink():
            raise ValueError('release_controller_symlink_forbidden:' + str(destination))
        if not verify_only:
            shutil.copyfile(source / name, destination)
        if not destination.is_file() or destination.read_bytes() != (source / name).read_bytes():
            raise ValueError('release_controller_source_mismatch:' + name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    sync(args.bundle, verify_only=args.verify_only)
    print('V282_RELEASE_CONTROLLERS=BYTE_IDENTICAL')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
