# Sound Protocol 0.1

Sound Protocol is a versioned description of observable field state. It carries
no audio samples, stimulus labels, field arrays, or renderer-specific controls.

Protocol 0.1 is experimental. Compatibility is guaranteed only for the exact
`0.1` schema.

## Frame identity

| Field | Unit | Range |
| --- | --- | --- |
| `version` | protocol revision | `"0.1"` |
| `sequence` | emitted frame index | integer, at least 0 |
| `step` | simulation step | integer, at least 0 |
| `time_seconds` | simulation time | seconds, at least 0 |

## Field state

Synchrony and metabolite values are dimensionless and lie in `[0, 1]`. Trace
values are dimensionless and non-negative.

## Region state

Every frame contains input, association, and output region snapshots calculated
with the same feature definitions.

Phase coherence and local synchrony are dimensionless values in `[0, 1]`.
Metabolite values are dimensionless values in `[0, 1]`. Trace and coupling are
dimensionless and non-negative. Frequency is measured in radians per simulation
second. Frequency and coupling spreads are population standard deviations.

## Pattern state

`PatternSnapshot` contains neutral observations of the output region. Phase
orders and phase-lock values lie in `[0, 1]`. Trace and frequency values are
non-negative. Metabolite stress and contrast lie in `[0, 1]`.

`ActivePattern` describes the currently accepted temporal pattern. Confidence,
intensity, and novelty lie in `[0, 1]`. Age is measured in emitted protocol
frames.

`PatternTransition` is present only when the accepted pattern starts, changes,
or ends. Continuous pattern state is carried separately in `active_pattern`.

## Cadence and detection

`frame_interval_steps` selects frames by absolute simulation step modulo the
configured interval. Pattern age and confirmation durations are measured in
emitted protocol frames.

`activity_threshold` and `confidence_threshold` decide whether an observation
can become a candidate. `confirmation_frames` controls temporal confirmation,
`minimum_active_frames` prevents premature replacement, `hysteresis_margin`
suppresses weak label changes, and `novelty_threshold` marks substantial changes
in the normalized pattern vector.

## Boundaries

The protocol does not contain the input stimulus label or any ground-truth
classification. Brightness, roughness, pitch, gain, rhythm, and other acoustic
controls belong to protocol consumers.

## JSON and replay

The JSON representation uses the field names from the data model and rejects
missing or unknown fields. Every document must declare version `0.1`. Numbers
must be finite JSON numbers; `NaN`, positive infinity, and negative infinity are
invalid.

JSONL recordings contain exactly one complete protocol frame per line. Blank
lines are invalid. A recording can be replayed repeatedly without constructing
or advancing a field simulation.
