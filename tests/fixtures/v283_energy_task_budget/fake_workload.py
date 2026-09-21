#!/usr/bin/env python3
import json
import os
from pathlib import Path
with Path(os.environ['R3_FAKE_EVENTS']).open('a') as f:
    f.write(json.dumps({'kind': 'workload', 'pid': os.getpid()}) + '\n')
print('__SPLITPOINT_WORK_UNITS__=100\n__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames\n__SPLITPOINT_WORK_UNITS_EXACT__=1')
