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

## Boundaries

The protocol does not contain the input stimulus label or any ground-truth
classification. Brightness, roughness, pitch, gain, rhythm, and other acoustic
controls belong to protocol consumers.
