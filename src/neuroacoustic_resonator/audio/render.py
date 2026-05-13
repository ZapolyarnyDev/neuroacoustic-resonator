from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from neuroacoustic_resonator.audio.output import ContinuousAudioRenderer, write_wav
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation


def default_audio_output_path(config_path: Path, steps: int) -> Path:
    return Path("experiments") / "audio" / f"{config_path.stem}-{steps}-steps.wav"


def steps_for_duration(
    duration_seconds: float,
    *,
    sample_rate: int,
    frame_size: int,
) -> int:
    if duration_seconds <= 0.0:
        msg = "duration_seconds must be positive"
        raise ValueError(msg)
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)

    return max(1, math.ceil(duration_seconds * sample_rate / frame_size))


def render_audio_demo(
    config_path: str | Path,
    *,
    steps: int | None = None,
    duration_seconds: float | None = None,
    sample_rate: int = 48_000,
    frame_size: int = 512,
    gain: float = 0.2,
    carrier_frequency: float = 220.0,
    frequency_scale: float = 1.0,
    output_path: str | Path | None = None,
) -> Path:
    if steps is not None and steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)
    if steps is not None and duration_seconds is not None:
        msg = "provide either steps or duration_seconds, not both"
        raise ValueError(msg)

    config_path = Path(config_path)
    config = SimulationConfig.from_file(config_path)
    total_steps = steps or (
        steps_for_duration(
            duration_seconds,
            sample_rate=sample_rate,
            frame_size=frame_size,
        )
        if duration_seconds is not None
        else config.steps
    )
    simulation = Simulation.from_config(config)
    regions = RegionMasks.from_size(config.field.size)
    renderer = ContinuousAudioRenderer(
        sample_rate=sample_rate,
        frame_size=frame_size,
        carrier_frequency=carrier_frequency,
        frequency_scale=frequency_scale,
        gain=gain,
    )
    frames = []

    for _ in range(total_steps):
        frame = simulation.step()
        frames.append(renderer.render_frame(frame.state, regions))

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float64)
    path = (
        Path(output_path)
        if output_path is not None
        else default_audio_output_path(
            config_path,
            total_steps,
        )
    )
    return write_wav(path, audio, sample_rate=sample_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an offline WAV demo from the output field region.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "default.yaml",
        help="YAML simulation config path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Target audio duration. Defaults to 5 seconds when --steps is omitted.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="WAV sample rate.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=512,
        help="Audio samples rendered per simulation step.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.2,
        help="Output gain before clipping.",
    )
    parser.add_argument(
        "--carrier-frequency",
        type=float,
        default=220.0,
        help="Audible carrier frequency for field frequency 1.0.",
    )
    parser.add_argument(
        "--frequency-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the carrier mapping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps is not None and args.duration_seconds is not None:
        msg = "provide either --steps or --duration-seconds, not both"
        raise ValueError(msg)

    duration_seconds = args.duration_seconds if args.steps is None else None
    if args.steps is None and duration_seconds is None:
        duration_seconds = 5.0

    if args.steps is None:
        assert duration_seconds is not None
        total_steps = steps_for_duration(
            float(duration_seconds),
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
        )
    else:
        total_steps = args.steps
    output_path = args.output or default_audio_output_path(args.config, total_steps)
    written_path = render_audio_demo(
        args.config,
        steps=total_steps,
        duration_seconds=None,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        gain=args.gain,
        carrier_frequency=args.carrier_frequency,
        frequency_scale=args.frequency_scale,
        output_path=output_path,
    )

    duration_seconds = total_steps * args.frame_size / args.sample_rate
    print(
        f"Rendered audio demo: {written_path} "
        f"steps={total_steps} duration={duration_seconds:.3f}s"
    )
    return 0
