from __future__ import annotations

from typing import Protocol

from neuroacoustic_resonator.analysis.protocol_features import (
    field_snapshot,
    pattern_snapshot,
    region_snapshot,
)
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import SimulationFrame
from neuroacoustic_resonator.protocol import (
    ActivePattern,
    PatternSnapshot,
    PatternTransition,
    SoundProtocolFrame,
)


class PatternDetector(Protocol):
    def update(
        self,
        snapshot: PatternSnapshot,
    ) -> tuple[ActivePattern | None, PatternTransition | None]: ...

    def reset(self) -> None: ...


class ProtocolEncoder:
    def __init__(
        self,
        *,
        dt: float,
        frame_interval_steps: int = 1,
        detector: PatternDetector | None = None,
    ) -> None:
        if dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if frame_interval_steps < 1:
            msg = "frame_interval_steps must be positive"
            raise ValueError(msg)
        self.dt = dt
        self.frame_interval_steps = frame_interval_steps
        self.detector = detector
        self._last_seen_step: int | None = None
        self._next_sequence = 0

    def encode(
        self,
        frame: SimulationFrame,
        regions: RegionMasks,
    ) -> SoundProtocolFrame | None:
        step = frame.metrics.step
        if self._last_seen_step is not None and step <= self._last_seen_step:
            msg = (
                "simulation step must increase strictly: "
                f"previous={self._last_seen_step}, current={step}"
            )
            raise ValueError(msg)
        self._last_seen_step = step

        if step % self.frame_interval_steps != 0:
            return None

        observed_pattern = pattern_snapshot(frame.state, regions.output)
        active_pattern, transition = (
            self.detector.update(observed_pattern)
            if self.detector is not None
            else (None, None)
        )
        encoded = SoundProtocolFrame(
            sequence=self._next_sequence,
            step=step,
            time_seconds=step * self.dt,
            field=field_snapshot(frame),
            input_region=region_snapshot(frame, regions.input, "input"),
            assoc_region=region_snapshot(frame, regions.assoc, "assoc"),
            output_region=region_snapshot(frame, regions.output, "output"),
            pattern=observed_pattern,
            active_pattern=active_pattern,
            transition=transition,
        )
        self._next_sequence += 1
        return encoded

    def reset(self) -> None:
        self._last_seen_step = None
        self._next_sequence = 0
        if self.detector is not None:
            self.detector.reset()
