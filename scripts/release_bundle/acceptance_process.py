#!/usr/bin/env python3
"""Finite owned-process supervision for the ordinary v2.81 acceptance runner."""
import os
from pathlib import Path
import signal
import subprocess
import time

def active_owned_group(group_id):
    """Inspect only the session/group created by this stage, excluding zombies."""
    active = []
    namespace = Path('/proc/self/ns/pid').stat().st_ino
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / 'ns/pid').stat().st_ino != namespace:
                continue
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            ids = {line.split(':', 1)[0]: line.split(':', 1)[1].split()
                   for line in (entry / 'status').read_text().splitlines()
                   if line.startswith(('NSpid:', 'NSpgid:', 'NSsid:'))}
            # /proc can be mounted from an ancestor PID namespace. Compare
            # logical IDs in our proven namespace, never unrelated host IDs.
            pgid = int((ids.get('NSpgid') or [fields[2]])[-1])
            sid = int((ids.get('NSsid') or [fields[3]])[-1])
            if fields[0] != 'Z' and pgid == group_id and sid == group_id:
                active.append(int((ids.get('NSpid') or [entry.name])[-1]))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(active)

def cleanup_owned_group(group_id):
    observed = active_owned_group(group_id)
    cleanup = {'owned_process_group': group_id, 'observed_live_pids': observed, 'status': 'clean'}
    # Shell log relays and Python resource trackers may finish just after their
    # parent. Give that ordinary exit a short bounded drain before escalation.
    deadline = time.monotonic() + 1.0
    while observed and time.monotonic() < deadline:
        time.sleep(.05)
        observed = active_owned_group(group_id)
    if not observed:
        return cleanup
    for sig, grace in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while active_owned_group(group_id) and time.monotonic() < deadline:
            time.sleep(.05)
        if not active_owned_group(group_id):
            break
    cleanup['remaining_live_pids'] = active_owned_group(group_id)
    cleanup['status'] = 'failed' if cleanup['remaining_live_pids'] else 'residual_owned_group_terminated'
    return cleanup

def run_stage(command, directory, log, environment, *, timeout_s=3600):
    started = last_output = time.monotonic()
    cancelled = None
    escalation = None
    timed_out = False
    with log.open('wb', buffering=0) as output, log.open('rb') as reader:
        child = subprocess.Popen(command, cwd=str(directory), stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
        def request_cancel(signum, frame):
            nonlocal cancelled
            if cancelled is None:
                cancelled = time.monotonic()
                print('CANCELLATION_REQUESTED=YES; waiting for owned stage cleanup', flush=True)
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
        previous = {sig: signal.signal(sig, request_cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while True:
                now = time.monotonic()
                if now - started >= timeout_s and cancelled is None:
                    timed_out = True
                    request_cancel(signal.SIGTERM, None)
                if cancelled is not None and child.poll() is None:
                    if now - cancelled >= 180 and escalation != 'kill':
                        os.killpg(child.pid, signal.SIGKILL); escalation = 'kill'
                    elif now - cancelled >= 160 and escalation is None:
                        os.killpg(child.pid, signal.SIGTERM); escalation = 'term'
                body = reader.read(65536)
                if body:
                    # Stream delegated output; final UPLOAD_ONLY names the one
                    # combined archive needed from the user.
                    text = body.decode('utf-8', errors='replace')
                    print(text, end='', flush=True)
                    last_output = time.monotonic()
                    # The leader may have exited while a descendant keeps the
                    # log busy. Retain the full log on disk and close our group.
                    if child.poll() is not None:
                        break
                elif child.poll() is not None:
                    break
                else:
                    now = time.monotonic()
                    if now - last_output >= 60:
                        print(f'STAGE_RUNNING=YES ELAPSED_S={int(now-started)} LIVE_LOG={log}', flush=True)
                        last_output = now
                    time.sleep(0.2)
            code = child.wait()
            cleanup = cleanup_owned_group(child.pid)
            if cleanup['status'] != 'clean' and code == 0:
                # Cleanup succeeded after finding a stage leak, but the stage
                # did not satisfy its own completion contract.
                code = 2
        except BaseException as primary:
            cleanup = {'owned_process_group': child.pid, 'status': 'not_needed'}
            if child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                        cleanup['status'] = 'terminated_after_controller_error'
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                        cleanup['status'] = 'killed_owned_group_after_controller_error'
                except Exception as error:
                    cleanup.update(status='failed', error=type(error).__name__ + ':' + str(error))
            try:
                cleanup['residual_group'] = cleanup_owned_group(child.pid)
            except Exception as error:
                cleanup.update(status='failed', error=type(error).__name__ + ':' + str(error))
            primary.stage_cleanup = cleanup
            raise
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
    return {'returncode': code, 'cancelled': cancelled is not None, 'timed_out': timed_out,
            'forced_termination': escalation, 'process_cleanup': cleanup,
            'elapsed_s': round(time.monotonic() - started, 3)}
