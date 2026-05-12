from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuroacoustic_resonator.audio_output import render_output_frame, write_wav
from neuroacoustic_resonator.config import SimulationConfig
from neuroacoustic_resonator.regions import RegionMasks
from neuroacoustic_resonator.simulation import Simulation


def default_audio_output_path(config_path: Path, steps: int) -> Path:
    return Path("experiments") / "audio" / f"{config_path.stem}-{steps}-steps.wav"


def render_audio_demo(
    config_path: str | Path,
    *,
    steps: int | None = None,
    sample_rate: int = 48_000,
    frame_size: int = 512,
    gain: float = 0.2,
    output_path: str | Path | None = None,
) -> Path:
    if steps is not None and steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)

    config_path = Path(config_path)
    config = SimulationConfig.from_file(config_path)
    total_steps = steps or config.steps
    simulation = Simulation.from_config(config)
    regions = RegionMasks.from_size(config.field.size)
    frames = []

    for _ in range(total_steps):
        frame = simulation.step()
        frames.append(
            render_output_frame(
                frame.state,
                regions,
                sample_rate=sample_rate,
                frame_size=frame_size,
                gain=gain,
            )
        )

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
        help="Number of simulation steps. Defaults to the config value.",
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
        "--output",
        type=Path,
        default=None,
        help="Output WAV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig.from_file(args.config)
    total_steps = args.steps or config.steps
    output_path = args.output or default_audio_output_path(args.config, total_steps)
    written_path = render_audio_demo(
        args.config,
        steps=total_steps,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        gain=args.gain,
        output_path=output_path,
    )

    duration_seconds = total_steps * args.frame_size / args.sample_rate
    print(
        f"Rendered audio demo: {written_path} "
        f"steps={total_steps} duration={duration_seconds:.3f}s"
    )
    return 0
