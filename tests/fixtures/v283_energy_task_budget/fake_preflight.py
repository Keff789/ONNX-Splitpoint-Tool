#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import time
events = Path(os.environ['R3_FAKE_EVENTS'])
with events.open('a') as f:
    f.write(json.dumps({'kind': 'preflight', 'pid': os.getpid()}) + '\n')
if os.environ.get('R3_PREFLIGHT_FAIL') == '1':
    raise SystemExit(5)
now = time.time_ns()
data = {'schema': 'onnx-splitpoint/energy-preflight-attestation',
        'schema_version': 1, 'ok': True, 'artifact_verification_status': 'pass',
        'nonce': sys.argv[1], 'command_contract_sha256': 'a' * 64,
        'created_at_unix_ns': now, 'expires_at_unix_ns': now + 30_000_000_000}
data['attestation_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
print(json.dumps(data))
