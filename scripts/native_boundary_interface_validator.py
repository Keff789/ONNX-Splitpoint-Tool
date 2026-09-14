#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

# Backwards-compatible alias for the contract validator script.
script = Path(__file__).with_name('native_boundary_interface_contract_validator.py')
if not script.exists():
    raise SystemExit(f'missing {script}')
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
