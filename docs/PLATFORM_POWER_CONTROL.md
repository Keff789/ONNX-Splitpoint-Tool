# u.RECS platform control and energy calibration

Release `2.79.20` retains unchanged the `2.79.19` separation of technical
validity from numerical plausibility in
the guided Full-System input gain calibration without changing the separate
direct M.2 idle-power measurement. The input-gain
routine is intended only for a remaining constant scale error of the already
linearized input measurement, including the real tolerance of the 20-mΩ input
shunt. It reuses each hardware setup's existing Jetson SSH data and
`energy.urecs_address`; no second host registry is introduced.

## Status model

Opening Tool Config for the first time schedules one bounded refresh for each of
the three setup cards (Hailo-8, Hailo-10 and DeepX). The Tool does not run a
periodic ping loop. Further checks occur only after pressing a card's **Refresh
status** button or completing a control operation.

The former global setup selector/status banner is not used. The three cards are
shown side by side and read the Jetson host, u.RECS address and accelerator-idle
value directly from the same central hardware registry as **Accelerator envs**.
The power cards do not save a second configuration copy before an operation, so
a stale card cannot overwrite a newer u.RECS address or calibration value.

The three displayed states have deliberately different evidence:

- **u.RECS host reachable**: one bounded ICMP probe to the configured host. It
  does not prove that UDP port 3000 or the controller command parser is ready;
- **Jetson ready**: an authenticated SSH connection using the setup's existing
  host, user, port and OpenSSH configuration;
- **M.2 accelerator enumerated**: a read-only probe on the Jetson for the
  configured Hailo or DeepX accelerator. Enumeration is not physical rail
  telemetry; missing enumeration may also mean a driver or probe failure.

Because the u.RECS firmware interface is toggle-only, an SSH-unreachable Jetson
is shown as unknown rather than asserted to be powered off. SSH-down is an
endpoint observation, not proof that the Jetson rail is off. Power actions from
an ambiguous observation require an explicit target and a fresh preflight.

## UDP control contract

### Explicit recovery after an unconfirmed capture end

An interrupted capture with `campaign_source_completion_unresolved` fences
its source, DUT, controller capture and controller NIC. Ending the processes
or observing an unheld kernel lock does not release these durable fences.
Historical successful captures do not confirm the failed attempt's source end.

The normal product provides an explicit operator command for this exact case:

```bash
python -m onnx_splitpoint_tool.remote.process_lease_cli recover-capture \
  --run-dir /absolute/path/to/failed-run \
  --session-id ORIGINAL_SESSION --operation-id ORIGINAL_CAPTURE \
  --setup-id REGISTRY_SETUP --source source:REGISTRY_SOURCE \
  --operator OPERATOR_NAME \
  --action 'Actual documented intervention already performed' \
  --performed-at 'ISO8601 timestamp with timezone after the STOP' \
  --supply-effect 'Actual effect on measurement controller and DUT supply' \
  --ready-observation 'Observed new readiness after the intervention' \
  --confirm-action-performed
```

These are placeholders, not evidence or a device-reset recipe. First establish
the actual device/operator procedure and its supply effects. The Tool does not
invent an Idle-ACK, reset the measurement controller, toggle rails, reboot a DUT,
or take a probe measurement. A desired release is not an intervention already
performed. If the procedure is not documented locally, obtain it from the
operator before acting. The `jetson`/`m.2` rail toggles below are not source-reset
commands. `--registry PATH` optionally selects the normal hardware registry.

The command binds the original capture request/reply and all four resource
identities to that registry. It holds every existing exclusive resource lock,
requires exact durable collector cleanup, a terminal parent and no unresolved
remote lease descriptors, then checks current local/remote ownership using the
existing read-only SSH transport. Active, foreign or unobservable predecessors,
wrong operations/resources, incomplete action evidence and changed metadata
remain blocked. No process is killed by this recovery command.

The original error and fence/owner evidence, actual operator action and current
ownership observations are retained in a `recovery` field of the existing
capture `resource-reply.json`. Its STOP/reason remain unchanged, as do the old
measurement, `finished=false`, retry/source budgets and acceptance counters.
No new journal or hash identity is created. All resource locks stay held until
the group transition is durably recorded; exceptions restore the original
fences, and an interrupted recovery writer remains fail-closed. Normal
admission never automatically clears a fence. Repeating the same successful
operator call returns `already_released` without starting work; a later foreign
quarantine cannot be cleared by replaying an earlier release.

This operator is limited to the physical capture STOP. It does not broaden
workflow Resume or establish general recovery from arbitrary controller crashes.

The default controller port is `3000`. The commands are configurable but
normally remain:

```text
jetson
m.2
```

The punctuation in `m.2` is part of the firmware command. Registries written
by releases 2.79.9--2.79.11 are migrated from the exact old default `m2` (or a
missing value) to `m.2`. Any other operator-defined command is preserved
exactly; the migration does not trim, case-fold or guess custom tokens.

The default line ending is LF, matching an interactive `nc -u HOST 3000`
session after pressing Enter. `none` and `crlf` are available for firmware
variants. The Tool sends one UDP datagram through Python's socket API, so a
local `nc` executable is not required. A successful `sendto()` has **no
application-level acknowledgement** and proves neither controller receipt nor
the resulting rail state. The commands are toggles, not idempotent set-state
operations.

Before a state-changing operation, the Tool validates the unique setup row,
effective Jetson SSH target, u.RECS address/port resolution, both exact ASCII
command tokens, terminator, safety booleans and shutdown/preflight commands.
It then reloads and compares the relevant registry projection after acquiring
the platform lock. Invalid, duplicated or changed configuration fails before
any shutdown is scheduled and before any UDP datagram is sent.

## Safety rules

State-changing operations are globally serialized across Tool processes and
also use the selected setup's operation lock. This is a Tool-level exclusion
contract; it does not claim that u.RECS exposes a physical controller identity
or controller-side locking. Operations are blocked by the global
Evaluation/power interlock while a workflow is active.
`power_control.enabled` is authoritative. The GUI also checks its own
background-job state before dispatching an operation. A command-line override
exists only for explicit maintainer recovery; normal operation is fail-closed.

**Set Jetson state** requires an explicit on/off target bound to a fresh SSH
observation. When Jetson SSH is ready and `off` is requested, it first verifies
non-interactive sudo, schedules a clean `systemctl poweroff` and waits until
SSH is down before a toggle is permitted. This reduces interruption risk but
does not turn SSH-down into independent rail-off telemetry. An `on` request
waits for authenticated SSH readiness and fails if the fresh state contradicts
the requested target.

**Toggle M.2** requires a known accelerator-presence state. It performs this
sequence:

1. cleanly shut down the Jetson and wait for SSH to disappear;
2. toggle the Jetson rail off;
3. toggle the M.2 rail;
4. toggle the Jetson rail on;
5. wait for SSH and verify the requested accelerator-presence state.

Unknown, stale or contradictory observations stop the sequence rather than
guessing. Accelerator-not-detected is never treated as a confirmed M.2-off
rail state.

## Full-System input gain calibration

Each setup card has a **Calibrate full-system input** button. The electronic
load must be connected as a sink from `9V_20V_IN` on the load side of R16 to
GND. The 5-V header is deliberately rejected as the instructed reference
point: a 0.5 A or 1.0 A load there passes through the 5-V converter and does
not produce the same input-current step at R16.

The routine requires a current status with a reachable u.RECS. An SSH-ready
Jetson is a known ON start. An SSH-unreachable Jetson is accepted only after the
operator explicitly confirms that it is deliberately powered off and stable;
SSH-down alone is not treated as rail telemetry. An unknown Jetson observation
still blocks the operation. M.2 enumeration is not a start gate because the
calibration never switches the M.2 rail.

The routine performs this sequence under the existing global
workflow/platform interlock:

1. require the operator to confirm that the correctly connected electronic
   load is off;
2. if the Jetson is initially SSH-ready, request one controlled Jetson-off
   transition; if it was explicitly confirmed already off, perform no hardware
   action;
3. measure `idle_before`;
4. show the nominal 0.5000 A setpoint, let the operator enter the electronic
   load's actual displayed current to four decimal places and its actual
   voltage, then measure `load_0.5A`;
5. require 0 A and measure `idle_between`;
6. repeat the separate nominal/actual entry for the 1.0000 A setpoint and
   measure `load_1A`;
7. require 0 A and measure `idle_after`;
8. calculate the factor, restore the recorded initial Jetson state and present
   the result for an explicit save decision. An initially ON Jetson receives one
   on transition; an initially OFF Jetson remains off. M.2 is untouched in both
   cases.

Each actual-current field is prefilled with its nominal value, 0.5000 A or
1.0000 A. The nominal setpoint remains a separate evidence field, but only the
operator-entered actual current is used in reference-power and fit
calculations. Cancelling or closing a load-point input dialog aborts that point
cleanly and enters the existing recovery path; it does not skip or reorder
`idle_before`, `idle_between` or `idle_after`.

Every one of the five capture phases uses the normal acquisition-integrity and
final-Energy gates. If an attempt is rejected for a transient acquisition
integrity condition, including nonzero dropped marker samples or a marker trace
that does not cover the command window, the collector may repeat that capture
once. This is a fresh acquisition with the same gates, not acceptance of the
failed sample and not a relaxed threshold. A second invalid attempt stops the
calibration. The operational JSON preserves the failed capture, raw measurement
summary, acquisition-integrity reasons and retry diagnostics. For load captures,
the previously entered nominal setpoint, actual current and actual voltage are
written before acquisition, so they remain available even when that capture
fails.

For the 0.5-A point the baseline is the mean of `idle_before` and
`idle_between`; for the 1-A point it is the mean of `idle_between` and
`idle_after`. With measured power increments `dP_i` and reference powers
`P_ref_i = I_actual_i * V_actual_i`, the one-factor fit through the origin is:

```text
factor = sum(dP_i * P_ref_i) / sum(dP_i²)
P_FS_corrected = factor * P_FS_measured
```

The review reports the maximum fit residual and two distinct classes of
diagnostics. Non-finite/non-positive input values, a singular fit, a factor
outside the broad hard safety range, insufficient measured increment and
excessive idle drift are technical blockers; they leave the registry unchanged.
Point-factor spread, the narrower configured expected-factor range and
reference-current deviation from the nominal setpoint are plausibility
warnings. They remain visible in the GUI and evidence but do not prevent an
otherwise technically valid result from being saved. The operator-entered
actual current and voltage still determine the fit; warnings neither replace
nor modify a measured value. Closing or cancelling the GUI while the recovery
prompt is active cannot skip the mandatory 0-A confirmation before automatic
restoration/check of the initial Jetson state. Load safety is independent of
whether a hardware toggle occurred: after either load dialog, an initially OFF
run also requires the explicit 0-A confirmation on failure. M.2 is never used as
a recovery action.

A successful save writes an immutable JSON evidence file, records its absolute
path and SHA-256 together with the factor and timestamp and clears existing
`idle_baseline_w`, `accelerator_idle_w` and M.2-idle evidence/binding fields.
Those values were measured in the old scale domain and must be recalibrated.
Normal FS acquisition verifies schema, setup identity, fit, captures, restore
state, file hash and registry binding before use. A configured but invalid
binding returns `full_system_current_scale_claim_blocked` before collector,
transport or workload start. MB and other non-FS scopes are not scaled.
Release 2.79.18 extended the existing evidence schema additively so that a
verified OFF-to-OFF restoration is valid; legacy ON/M.2-present evidence remains
readable. Release 2.79.19 adds only the warning-reason projection to that
evidence; it introduces no additional hash, manifest, signature or sealing
layer.

## Idle-power calibration

The calibration starts only with an SSH-ready Jetson and the configured M.2
accelerator positively detected. The default windows are 30 seconds of
stabilization followed by 30 seconds of u.RECS measurement per state.

The Tool first requests the M.2-off transition and verifies repeated,
post-boot Jetson endpoint and accelerator-enumeration observations. It probes
the requested state immediately before and immediately after the first capture.
It subsequently requests M.2-on, repeats the bounded post-boot observations,
and performs the same pre- and post-capture state gates around the second
measurement. A changed or ambiguous observation invalidates that capture.
The calibrated value is

```text
accelerator_idle_w = mean_power_with_m2 - mean_power_without_m2
```

The existing `energy.accelerator_idle_w` value remains untouched until both
captures have succeeded, the difference passes the configured plausibility
check, and the final enumeration observation is M.2-present. Calibration
revalidates its setup configuration after acquiring the platform lock and uses
a compare-and-swap commit against that exact snapshot. A concurrent edit is
reported as stale state instead of being overwritten. A readable
`m2_idle_power_calibration.json` and the ordinary off/on measurement
directories are retained below the energy-measurement calibration root. The
result file contains the two Full-System means, their difference, timestamps
and the observed states. It is not sealed and carries no calibration SHA.

This is a verified endpoint/enumeration calibration over an unacknowledged
toggle interface. It is **not** a claim that u.RECS exposes hardware-safe,
acknowledged set-state control. Standalone power operations do not have a
restartable transaction journal; after process termination or uncertain UDP
delivery, the operator must refresh status and recover the platform explicitly
before another toggle.

The calibrated value is applied only to an exact TensorRT Full role
(`source_run_id=native_full_tensorrt`, `target_variant=full`). The Tool accepts
the value only when the result JSON still matches its setup, accelerator,
u.RECS address, data port, timestamps and `on - off` arithmetic. This is a
plain consistency check, not a hash chain. The raw measured energy is always
preserved next to the normalized comparison value. Hailo, DeepX,
split-pipeline and calibration measurements are never corrected by this value.

## Registry keys

Each hardware setup may contain a `power_control` mapping. Existing registries
are migrated additively to schema version 2; user overrides are preserved.
Important defaults are:

```yaml
power_control:
  enabled: true
  udp_port: 3000
  udp_terminator: lf
  jetson_command: jetson
  m2_command: m.2
  status_ping_timeout_s: 1.5
  ssh_probe_timeout_s: 5.0
  m2_probe_timeout_s: 12.0
  shutdown_timeout_s: 120.0
  boot_timeout_s: 240.0
  m2_post_boot_settle_s: 5.0
  m2_verify_observations: 2
  m2_verify_interval_s: 1.0
  calibration_stabilize_s: 30.0
  calibration_measure_s: 30.0
  full_system_calibration_load_settle_s: 5.0
  full_system_calibration_minimum_delta_w: 2.0
  full_system_calibration_max_point_spread_pct: 2.0
  full_system_calibration_max_idle_drift_w: 0.5
  full_system_calibration_min_factor: 0.90
  full_system_calibration_max_factor: 1.10
  full_system_calibration_reference_current_tolerance_pct: 10.0
  require_ping_before_toggle: true
  require_positive_calibration_delta: true
  minimum_calibration_delta_w: 0.02
```

One successful Full-System input calibration adds these fields to the selected
setup:

```yaml
energy:
  full_system_current_scale_factor: 1.0062
  full_system_current_scale_calibrated_at: "2026-09-03T12:30:00+00:00"
  full_system_current_scale_calibration_evidence: /absolute/path/to/full_system_input_scale_calibration.json
  full_system_current_scale_calibration_sha256: 64-lowercase-hex-digits
```

A subsequent successful M.2 idle calibration adds these fields:

```yaml
energy:
  accelerator_idle_w: 1.6
  accelerator_idle_calibrated_at: "2026-09-03T12:34:56+00:00"
  accelerator_idle_calibration_evidence: /absolute/path/to/m2_idle_power_calibration.json
```

`accelerator_idle_w` is displayed read-only in the GUI and is replaced only by
a complete calibration. The registry update is atomic. Older binding path/SHA
fields are removed when the new value is saved. Existing method-manifest fields
may remain for other energy workflows, but they are ignored by this M.2 idle
calibration.

## Command-line interface

The same backend used by the GUI is available through:

```bash
onnx-splitpoint-platform-power list
onnx-splitpoint-platform-power status orin_nx_hailo8_01
onnx-splitpoint-platform-power set-jetson orin_nx_hailo8_01 off
onnx-splitpoint-platform-power set-jetson orin_nx_hailo8_01 on
onnx-splitpoint-platform-power toggle-m2 orin_nx_hailo8_01
onnx-splitpoint-platform-power set-m2 orin_nx_hailo8_01 on
onnx-splitpoint-platform-power calibrate-m2-idle orin_nx_hailo8_01
```

Hardware-changing commands must not be used while an evaluation run is active.
The currently running campaign should therefore finish before installing this
release or exercising the new controls.

The source updater first takes an exclusive, nonblocking FD lock on the
canonical
`~/.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock` and retains it
through archive verification, replacement and final installed-source checks.
The current seven-model launcher takes the shared side before reading installed
release inputs; its detached worker inherits and retains that shared lock until
the workflow exits. The foreground-to-worker handoff therefore has no unlocked
race window. A running workflow blocks the updater before archive extraction,
and a running updater blocks new launch admission.

The updater also checks the actual flock state of the other
`~/.onnx_splitpoint_tool/locks/*.lock` files. Stale, unheld files do not block an
update; a held lock does. `--maintainer-allow-active-workflow` remains available
only for explicit recovery from those legacy/per-workflow locks and prints a
warning. It cannot bypass the canonical exclusive/shared interlock and must not
be used for a routine installation.
