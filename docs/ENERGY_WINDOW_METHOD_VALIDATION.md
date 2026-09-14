# Command-window versus historical flank-window validation

The Chapter-4 calibration work and the new Final command-window protocol solve
different parts of the measurement problem.  The existing electrical
calibration equations, environment correction and `power_calculations`
conversion remain unchanged.  The v2 protocol changes only how the samples
belonging to one workload command are selected.  Chapter 4 therefore continues
to describe its measurements with the historical flank/duration method; this
software comparison does not rewrite those results.

To quantify the effect instead of assuming it, the tool processes each v2 trace
twice:

1. The primary invocation uses `--command-window-request` and
   `--require-command-window`.  It integrates the collector's inclusive sample
   bounds and remains the only result eligible for Final gating.
2. The validation invocation reads the same measurement directory with the
   historical `-c -r --estimated-duration=<duration+2>` contract.  It therefore
   retains the power-edge detection and duration adaptation used by earlier
   measurements.  Its output is written below `processed_legacy_window/` and is
   diagnostic only.

The raw trace hash is checked before and after the validation invocation.
`window_method_comparison.json` records both command lines and hashes, both
result hashes, energy, duration and average power, the selected start/end
indices, marker uncertainty and drops, and signed legacy-minus-command-window
deltas in samples, milliseconds and percent.  It also reports a symmetric
percentage difference and `ln(E_command-window/E_legacy)` so method agreement
can be evaluated without privileging one value as the percentage denominator.
The artefact verifies that all non-window calibration/filter command settings
are identical and explicitly sets:

```json
{
  "diagnostic_only": true,
  "eligible_for_final_energy": false,
  "affects_primary_result": false,
  "affects_final_gate": false,
  "scientific_method_decision": "frozen_command_marker_primary_chapter4_shadow",
  "scientific_primary_method_frozen": true,
  "scientific_primary_method": "command_marker_window",
  "scientific_shadow_method": "chapter4_legacy_window"
}
```

Consequently, a failed or numerically different legacy comparison cannot
replace the marker result, relax a gate, or change energy-per-work-unit.
The aggregate `window_method_comparison_statistics` applies the campaign's
Student-t summary to absolute, relative, symmetric and log-energy differences,
duration and average-power differences, and start/end shifts.  It intentionally
does not apply an automatic `+/-1%` or other agreement gate.  Such a threshold
would first have to be justified and frozen in the validation protocol.

## Configuration

The comparison is enabled by default for the validation phase and can be
controlled in `energy_defaults`:

```yaml
energy_defaults:
  compare_legacy_window: true
```

The manual measurement CLI also accepts `--compare-legacy-window` and
`--no-compare-legacy-window`.  Disabling it saves postprocessing time but does
not change acquisition or the primary command-window result.

## Acceptance use and v2.67 decision addendum

Version 2.67 freezes `command_marker_window` as the prospective scientific
primary for the benchmark campaign. The choice is based on its explicit,
hash-bound command boundaries and is not an assertion that every marker value
must be numerically closer to the historical result. The Chapter-4
flank/duration method remains `chapter4_legacy_window`, a same-trace sensitivity
shadow that preserves continuity with the calibration work. Report its energy
and boundary deltas when available. Large or workload-dependent differences are
a method-sensitivity result to discuss; they are not evidence that the
electrical calibration changed.

A missing or failed Chapter-4 shadow never invalidates, reacquires, replaces or
changes an otherwise valid marker-primary repeat. No automatic method switch is
permitted. Archived v2.63--v2.66 artefacts that explicitly name
`chapter4_baseline` as their frozen primary retain that historical meaning when
read; the new role assignment applies only to role-explicit v2.67 artefacts.

The separate `0.97` correction factor is not re-estimated in this test.  It
remains frozen and is applied identically to both branches, so it cannot create
their relative window-method difference.  No new PicoScope measurement or
cross-instrument marker study is part of this validation plan.

## 2.66 probe execution correction

The separate screening probe owns its three independent repetitions. Each
outer repetition starts exactly one collector capture and receives a unique,
nonce-bound remote output root. The collector's normal A/B minimum-repeat
expansion remains active for ordinary energy rows but is explicitly disabled
inside this caller-managed probe loop. Collector-internal invalid-repeat retries
and in-repeat reconnect recovery are disabled: a failed trace is retained and
the probe stops fail-fast instead of starting a second physical capture for the
same outer repeat.

Probe strictness and workflow blocking are recorded separately. Since v2.67 an
incomplete strict probe is retained as a validation failure and warning but is
non-blocking in every run mode, including Final. Smoke, Standard and Final runs
therefore retain otherwise complete Native and marker-primary energy evidence
while making an incomplete Chapter-4 sensitivity comparison plainly visible.
