from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.protocol_stream import ProtocolAnalysisStream
from neuroacoustic_resonator.audio.io import write_wav
from neuroacoustic_resonator.audio.output import ProtocolReferenceRenderer
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.protocol.codec import (
    ProtocolReplay,
    write_protocol_jsonl,
)
from neuroacoustic_resonator.protocol.consumer import (
    CaptureConsumer,
    ProtocolConsumer,
    ProtocolPipeline,
)
from neuroacoustic_resonator.protocol.model import SoundProtocolFrame


@dataclass(frozen=True, slots=True)
class ProtocolRecordResult:
    path: Path
    frames: tuple[SoundProtocolFrame, ...]

    def summary(self) -> dict[str, Any]:
        return protocol_stream_summary(self.frames, source=str(self.path))


@dataclass(frozen=True, slots=True)
class ProtocolReplayResult:
    source: Path
    frames: tuple[SoundProtocolFrame, ...]

    def summary(self) -> dict[str, Any]:
        return protocol_stream_summary(self.frames, source=str(self.source))


def record_protocol(
    config_path: str | Path,
    output_path: str | Path,
    *,
    steps: int,
) -> ProtocolRecordResult:
    if steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    config = SimulationConfig.from_file(config_path)
    simulation = config.create_simulation()
    regions = RegionMasks.from_size(config.field.size)
    stream = ProtocolAnalysisStream.from_config(config, regions)
    stream.observe(
        simulation.snapshot(),
        input_value=simulation.last_input_value,
    )
    for _ in range(steps):
        stream.observe(
            simulation.step(),
            input_value=simulation.last_input_value,
        )
    destination = write_protocol_jsonl(output_path, stream.frames)
    return ProtocolRecordResult(destination, tuple(stream.frames))


def replay_protocol(
    input_path: str | Path,
    consumers: Iterable[ProtocolConsumer] = (),
) -> ProtocolReplayResult:
    source = Path(input_path)
    capture = CaptureConsumer()
    pipeline = ProtocolPipeline([capture, *consumers])
    pipeline.run(ProtocolReplay(source))
    return ProtocolReplayResult(source, tuple(capture.frames))


def protocol_stream_summary(
    frames: Iterable[SoundProtocolFrame],
    *,
    source: str,
) -> dict[str, Any]:
    materialized = tuple(frames)
    labels: dict[str, int] = {}
    transitions: dict[str, int] = {}
    for frame in materialized:
        label = "idle" if frame.active_pattern is None else frame.active_pattern.label
        labels[label] = labels.get(label, 0) + 1
        if frame.transition is not None:
            kind = frame.transition.kind
            transitions[kind] = transitions.get(kind, 0) + 1
    return {
        "source": source,
        "version": None if not materialized else materialized[0].version,
        "frames": len(materialized),
        "first_sequence": None if not materialized else materialized[0].sequence,
        "last_sequence": None if not materialized else materialized[-1].sequence,
        "first_step": None if not materialized else materialized[0].step,
        "last_step": None if not materialized else materialized[-1].step,
        "duration_seconds": (
            0.0
            if len(materialized) < 2
            else materialized[-1].time_seconds - materialized[0].time_seconds
        ),
        "pattern_counts": labels,
        "transition_counts": transitions,
    }


def render_protocol_audio(
    frames: Iterable[SoundProtocolFrame],
    output_path: str | Path,
    *,
    sample_rate: int = 48_000,
    frame_size: int = 512,
) -> Path:
    renderer = ProtocolReferenceRenderer(
        sample_rate=sample_rate,
        frame_size=frame_size,
    )
    rendered = [renderer.render_frame(frame) for frame in frames]
    audio = np.concatenate(rendered) if rendered else np.zeros(0, dtype=np.float64)
    return write_wav(output_path, audio, sample_rate=sample_rate)


def write_protocol_summary(
    path: str | Path,
    summary: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and replay Neuroacoustic Sound Protocol v0 streams.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="Run a simulation and write JSONL.")
    record.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "field_only.yaml",
    )
    record.add_argument("--steps", type=int, default=128)
    record.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "protocol" / "recording.jsonl",
    )
    record.add_argument("--summary", type=Path, default=None)

    replay = commands.add_parser("replay", help="Replay JSONL without a field.")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--summary", type=Path, default=None)
    replay.add_argument("--output-wav", type=Path, default=None)
    replay.add_argument("--sample-rate", type=int, default=48_000)
    replay.add_argument("--frame-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record":
        record_result = record_protocol(args.config, args.output, steps=args.steps)
        summary = record_result.summary()
        if args.summary is not None:
            write_protocol_summary(args.summary, summary)
        print(
            f"Recorded {summary['frames']} protocol frames "
            f"through step {summary['last_step']}: {record_result.path}"
        )
        return 0

    replay_result = replay_protocol(args.input)
    summary = replay_result.summary()
    if args.summary is not None:
        write_protocol_summary(args.summary, summary)
    if args.output_wav is not None:
        render_protocol_audio(
            replay_result.frames,
            args.output_wav,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
        )
    print(
        f"Replayed {summary['frames']} protocol frames "
        f"without a field: {replay_result.source}"
    )
    return 0
