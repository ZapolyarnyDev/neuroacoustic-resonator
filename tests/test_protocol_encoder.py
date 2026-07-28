from dataclasses import replace

import numpy as np
import pytest

from neuroacoustic_resonator.core.field import FieldConfig, OscillatorField
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import SimulationFrame
from neuroacoustic_resonator.encoding import ProtocolEncoder
from neuroacoustic_resonator.protocol import (
    ActivePattern,
    PatternSnapshot,
    PatternTransition,
)


class RecordingDetector:
    def __init__(self) -> None:
        self.snapshots: list[PatternSnapshot] = []
        self.reset_count = 0

    def update(
        self,
        snapshot: PatternSnapshot,
    ) -> tuple[ActivePattern | None, PatternTransition | None]:
        self.snapshots.append(snapshot)
        return (
            ActivePattern("coherent", 0.8, 0.5, 0.1, False, 1),
            PatternTransition("started", None, "coherent"),
        )

    def reset(self) -> None:
        self.reset_count += 1


def frame_at_step(step: int, *, size: int = 8) -> SimulationFrame:
    field = OscillatorField(FieldConfig(size=size, seed=1))
    return SimulationFrame(
        state=field.state,
        metrics=field.metrics(step=step),
        local_synchrony=field.local_synchrony(),
    )


def test_protocol_encoder_produces_deterministic_identity_and_time() -> None:
    encoder = ProtocolEncoder(dt=0.02)
    regions = RegionMasks.from_size(8)

    first = encoder.encode(frame_at_step(0), regions)
    second = encoder.encode(frame_at_step(1), regions)

    assert first is not None
    assert second is not None
    assert (first.sequence, first.step, first.time_seconds) == (0, 0, 0.0)
    assert (second.sequence, second.step, second.time_seconds) == (1, 1, 0.02)


def test_protocol_encoder_uses_absolute_modulo_cadence() -> None:
    encoder = ProtocolEncoder(dt=0.02, frame_interval_steps=4)
    regions = RegionMasks.from_size(8)

    skipped = encoder.encode(frame_at_step(3), regions)
    emitted = encoder.encode(frame_at_step(4), regions)

    assert skipped is None
    assert emitted is not None
    assert emitted.sequence == 0
    assert emitted.step == 4


def test_protocol_encoder_rejects_backwards_skipped_steps() -> None:
    encoder = ProtocolEncoder(dt=0.02, frame_interval_steps=4)
    regions = RegionMasks.from_size(8)
    encoder.encode(frame_at_step(4), regions)
    encoder.encode(frame_at_step(7), regions)

    with pytest.raises(ValueError, match="previous=7, current=6"):
        encoder.encode(frame_at_step(6), regions)


def test_protocol_encoder_accepts_structural_detector_and_resets_it() -> None:
    detector = RecordingDetector()
    encoder = ProtocolEncoder(dt=0.02, detector=detector)
    regions = RegionMasks.from_size(8)

    encoded = encoder.encode(frame_at_step(1), regions)
    encoder.reset()
    restarted = encoder.encode(frame_at_step(1), regions)

    assert encoded is not None
    assert encoded.active_pattern is not None
    assert encoded.transition == PatternTransition("started", None, "coherent")
    assert len(detector.snapshots) == 2
    assert detector.reset_count == 1
    assert restarted is not None
    assert restarted.sequence == 0


def test_protocol_encoder_rejects_region_shape_mismatch() -> None:
    encoder = ProtocolEncoder(dt=0.02)
    regions = RegionMasks.from_size(8)
    mismatched = replace(
        frame_at_step(1),
        local_synchrony=np.ones((7, 7)),
    )

    with pytest.raises(ValueError, match="local synchrony"):
        encoder.encode(mismatched, regions)
