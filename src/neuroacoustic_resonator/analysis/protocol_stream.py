from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from neuroacoustic_resonator.analysis.metrics import (
    ProtocolActivityTracker,
    RegionalActivityMetrics,
)
from neuroacoustic_resonator.analysis.pattern_detector import TemporalPatternDetector
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import SimulationFrame
from neuroacoustic_resonator.encoding import ProtocolEncoder
from neuroacoustic_resonator.protocol import (
    SoundProtocolFrame,
    write_protocol_jsonl,
)


@dataclass(frozen=True, slots=True)
class ProtocolObservation:
    frame: SoundProtocolFrame
    activity: RegionalActivityMetrics


class ProtocolAnalysisStream:
    def __init__(
        self,
        encoder: ProtocolEncoder,
        regions: RegionMasks,
        *,
        activity: ProtocolActivityTracker | None = None,
    ) -> None:
        self.encoder = encoder
        self.regions = regions
        self.activity = activity or ProtocolActivityTracker()
        self.frames: list[SoundProtocolFrame] = []
        self.last_observation: ProtocolObservation | None = None

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        regions: RegionMasks,
    ) -> ProtocolAnalysisStream:
        return cls(
            ProtocolEncoder(
                dt=config.field.dt,
                config=config.protocol.to_encoder_config(),
                detector=TemporalPatternDetector(config.protocol.to_detector_config()),
            ),
            regions,
        )

    @property
    def last_frame(self) -> SoundProtocolFrame:
        if not self.frames:
            msg = "protocol stream has no frames"
            raise ValueError(msg)
        return self.frames[-1]

    def observe(
        self,
        simulation_frame: SimulationFrame,
        *,
        input_value: float = 0.0,
    ) -> ProtocolObservation | None:
        frame = self.encoder.encode(simulation_frame, self.regions)
        if frame is None:
            return None
        self.frames.append(frame)
        observation = ProtocolObservation(
            frame=frame,
            activity=self.activity.update(frame, input_value=input_value),
        )
        self.last_observation = observation
        return observation


class ProtocolFrameHistory:
    def __init__(self, frames: Iterable[SoundProtocolFrame] = ()) -> None:
        self._frames = list(frames)

    def __iter__(self) -> Iterator[SoundProtocolFrame]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def append(self, frame: SoundProtocolFrame) -> None:
        self._frames.append(frame)

    def latest(self) -> SoundProtocolFrame:
        if not self._frames:
            msg = "protocol history is empty"
            raise ValueError(msg)
        return self._frames[-1]

    def to_rows(self) -> list[dict[str, Any]]:
        return [protocol_frame_row(frame) for frame in self._frames]

    def write_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            output.write_text("", encoding="utf-8")
            return output
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return output

    def write_jsonl(self, path: str | Path) -> Path:
        return write_protocol_jsonl(path, self._frames)


def protocol_frame_row(frame: SoundProtocolFrame) -> dict[str, Any]:
    active = frame.active_pattern
    transition = frame.transition
    row: dict[str, Any] = {
        "version": frame.version,
        "sequence": frame.sequence,
        "step": frame.step,
        "time_seconds": frame.time_seconds,
        "global_synchrony": frame.field.global_synchrony,
        "mean_local_synchrony": frame.field.mean_local_synchrony,
        "mean_metabolite": frame.field.mean_metabolite,
        "min_metabolite": frame.field.min_metabolite,
        "mean_trace": frame.field.mean_trace,
        "max_trace": frame.field.max_trace,
        "active_pattern_label": None if active is None else active.label,
        "active_pattern_confidence": 0.0 if active is None else active.confidence,
        "active_pattern_intensity": 0.0 if active is None else active.intensity,
        "active_pattern_novelty": 0.0 if active is None else active.novelty,
        "transition_kind": None if transition is None else transition.kind,
    }
    for prefix, region in (
        ("input", frame.input_region),
        ("assoc", frame.assoc_region),
        ("output", frame.output_region),
    ):
        row.update(
            {
                f"{prefix}_phase_coherence": region.phase_coherence,
                f"{prefix}_local_synchrony": region.mean_local_synchrony,
                f"{prefix}_metabolite": region.mean_metabolite,
                f"{prefix}_trace": region.mean_trace,
                f"{prefix}_frequency": region.mean_frequency,
                f"{prefix}_coupling": region.mean_coupling,
            }
        )
    for model_field in fields(frame.pattern):
        row[f"pattern_{model_field.name}"] = getattr(
            frame.pattern,
            model_field.name,
        )
    return row
