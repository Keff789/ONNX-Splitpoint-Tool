"""Opt-in loopback regressions against the exact R9L reviewed collector.

Set R9L_REVIEWED_COLLECTOR to the installed r6-reviewed binary. These tests
never build a collector, contact a device, or produce physical energy evidence.
The fake workload is a short, owned Python subprocess; output belongs under
pytest's external --basetemp. Timings retain the binary's real five-second
silence limit and completion grace, without changing measurement policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest


REVIEWED_SHA256 = "913c3f745a71809d85c1e98f56d0ca3265bb5e5492ad2b72407f7ce9bd8c3a46"
END_PACKET = b"Finished Measurement!\n"
REQUESTED_SECONDS = 0.6
COMPLETION_GRACE_SECONDS = 5.0


@pytest.fixture(scope="module")
def reviewed_collector():
    configured = os.environ.get("R9L_REVIEWED_COLLECTOR")
    if not configured:
        pytest.skip("opt-in requires the installed R9L reviewed collector; no build")
    binary = Path(configured).resolve(strict=True)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == REVIEWED_SHA256
    return binary


def _run_loopback(binary, output, *, end_after, stop_samples_after=None):
    """Only our bound loopback source receives GO; all counters are contiguous."""
    assert output.is_absolute()
    output.mkdir(parents=True, exist_ok=True)
    source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    source.bind(("127.0.0.1", 0))
    source.settimeout(3.0)
    done = threading.Event()
    observed = {}
    errors = []

    def serve():
        try:
            command, peer = source.recvfrom(1024)
            observed["go"] = command.decode("ascii")
            assert command == b"go 0 600000 2000\n"
            start = time.monotonic()
            observed["go_received_monotonic_ns"] = time.monotonic_ns()
            counter = 0
            next_packet = start
            sample_limit = end_after if stop_samples_after is None else stop_samples_after
            while not done.is_set():
                elapsed = time.monotonic() - start
                if end_after is not None and elapsed >= end_after:
                    source.sendto(END_PACKET, peer)
                    observed["end_sent_after_s"] = time.monotonic() - start
                    break
                if sample_limit is None or elapsed < sample_limit:
                    if time.monotonic() >= next_packet:
                        payload = b"".join(
                            struct.pack("<HH", (counter + offset) % 65536, 1000)
                            for offset in range(64)
                        )
                        source.sendto(payload, peer)
                        counter += 64
                        next_packet += 0.032  # 64 samples at the unchanged 2000 Hz.
                done.wait(0.001)
            observed["samples_sent"] = counter
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, name="r9l-loopback-source", daemon=True)
    workload = output / "fake_workload.py"
    workload.write_text(
        "from pathlib import Path\nimport time\n"
        "time.sleep(0.08)\n"
        f"Path({str(output / 'workload_completed.txt')!r}).write_text('completed\\n')\n"
    )
    command = [
        str(binary), "--storage-path", str(output), "--duration", "600ms",
        "--pre-duration", "0s", "--post-duration", "0s",
        "--command", shlex.join([sys.executable, str(workload)]),
        "--run-id", "r9l-loopback-only", "--window-id", output.name,
        "fast-firmware", "--address", "127.0.0.1", "--data-port",
        str(source.getsockname()[1]), "--channel", "0", "--sample-rate", "2000",
    ]
    environment = dict(os.environ, URECS_RECEIVE_DIAGNOSTICS="1")
    worker.start()
    try:
        result = subprocess.run(
            command, cwd=output, env=environment, capture_output=True,
            text=True, timeout=12, check=False,
        )
    finally:
        done.set()
        worker.join(timeout=4)
        source.close()
    assert not worker.is_alive(), "owned fake source did not stop"
    assert not errors, errors
    (output / "collector.stdout.log").write_text(result.stdout)
    (output / "collector.stderr.log").write_text(result.stderr)
    (output / "fake_source.json").write_text(json.dumps(observed, indent=2))
    diagnostics = [json.loads(line) for line in
                   (output / "receive_diagnostics.jsonl").read_text().splitlines()]
    events = {row["event"]: row for row in diagnostics if row["event"] != "datagram"}
    received = [row for row in diagnostics if row["event"] == "datagram"
                and row["details"]["received_length"] == 256]
    assert received
    assert events["receive_finished"]["details"]["dropped_samples"] == 0
    assert events["writer_close_end"]["details"]["ok"] is True
    assert events["writer_close_end"]["monotonic_ns"] <= events["socket_closed"]["monotonic_ns"]
    assert (output / "workload_completed.txt").read_text() == "completed\n"
    parquet, = output.glob("*.parquet")
    assert parquet.read_bytes()[-4:] == b"PAR1", "receive failure must still close the writer"
    return result, events, diagnostics


@pytest.mark.parametrize("end_after", [0.8, REQUESTED_SECONDS + 4.4],
                         ids=["normal-end", "late-end-within-completion-grace"])
def test_exact_end_within_protocol_window_closes_and_succeeds(reviewed_collector, tmp_path, end_after):
    result, events, _ = _run_loopback(reviewed_collector, tmp_path, end_after=end_after)
    assert result.returncode == 0, result.stderr
    assert events["receive_finished"]["details"]["ok"] is True
    assert events["socket_closed"]["details"]["outcome_ok"] is True
    assert "protocol_end_verified" in events
    assert events["exact_end_datagram"]["details"]["elapsed_us"] >= 600000
    assert (tmp_path / "command_window_markers.json").exists()


def test_contiguous_stream_then_silence_retains_failure_and_closed_writer(reviewed_collector, tmp_path):
    result, events, diagnostics = _run_loopback(
        reviewed_collector, tmp_path, end_after=None, stop_samples_after=0.3,
    )
    assert result.returncode == 1
    reason = events["receive_finished"]["details"]["error"]
    assert "silence timeout; device completion unverified" in reason
    assert "WouldBlock" in reason or "TimedOut" in reason
    assert reason in result.stderr
    assert events["socket_closed"]["details"]["outcome_ok"] is False
    assert "protocol_end_verified" not in events
    assert "exact_end_datagram" not in events
    last_packet = max(row["monotonic_ns"] for row in diagnostics if row["event"] == "datagram")
    assert (events["receive_finished"]["monotonic_ns"] - last_packet) / 1e9 >= 5.0
    assert not (tmp_path / "command_window_markers.json").exists()


def test_premature_exact_end_is_not_verified_completion(reviewed_collector, tmp_path):
    result, events, _ = _run_loopback(reviewed_collector, tmp_path, end_after=0.25)
    assert result.returncode == 1
    reason = events["receive_finished"]["details"]["error"]
    assert "premature end packet" in reason
    assert reason in result.stderr
    assert events["exact_end_datagram"]["details"]["elapsed_us"] < 600000
    assert events["socket_closed"]["details"]["outcome_ok"] is False
    assert "protocol_end_verified" not in events
    assert not (tmp_path / "command_window_markers.json").exists()


def test_end_beyond_absolute_deadline_cannot_rescue_continuous_stream(reviewed_collector, tmp_path):
    result, events, _ = _run_loopback(
        reviewed_collector, tmp_path,
        end_after=REQUESTED_SECONDS + COMPLETION_GRACE_SECONDS + 0.5,
    )
    assert result.returncode == 1
    reason = events["receive_finished"]["details"]["error"]
    assert "missing end packet at absolute deadline" in reason
    assert reason in result.stderr
    assert events["socket_closed"]["details"]["outcome_ok"] is False
    assert "protocol_end_verified" not in events
    assert not (tmp_path / "command_window_markers.json").exists()
