from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.audio_output import ContinuousAudioRenderer
from neuroacoustic_resonator.config import SimulationConfig
from neuroacoustic_resonator.regions import RegionMasks
from neuroacoustic_resonator.simulation import Simulation

FloatArray = NDArray[np.float64]
OutputCallback = Callable[[NDArray[np.float32], int, Any, Any], None]


class OutputStreamLike(Protocol):
    def __enter__(self) -> OutputStreamLike: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


StreamFactory = Callable[..., OutputStreamLike]


@dataclass(frozen=True)
class RealtimeAudioConfig:
    config_path: Path = Path("configs") / "audio_demo.yaml"
    sample_rate: int = 48_000
    frame_size: int = 512
    gain: float = 0.2
    carrier_frequency: float = 220.0
    frequency_scale: float = 1.0
    smoothing: float = 0.2
    physics_steps_per_audio_frame: int = 1
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if self.frame_size < 1:
            msg = "frame_size must be positive"
            raise ValueError(msg)
        if self.gain < 0.0:
            msg = "gain must be non-negative"
            raise ValueError(msg)
        if self.carrier_frequency <= 0.0:
            msg = "carrier_frequency must be positive"
            raise ValueError(msg)
        if self.frequency_scale <= 0.0:
            msg = "frequency_scale must be positive"
            raise ValueError(msg)
        if not 0.0 < self.smoothing <= 1.0:
            msg = "smoothing must be in (0, 1]"
            raise ValueError(msg)
        if self.physics_steps_per_audio_frame < 1:
            msg = "physics_steps_per_audio_frame must be positive"
            raise ValueError(msg)
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            msg = "duration_seconds must be positive"
            raise ValueError(msg)


class RealtimeAudioEngine:
    def __init__(self, config: RealtimeAudioConfig) -> None:
        self.config = config
        simulation_config = SimulationConfig.from_file(config.config_path)
        self.simulation = Simulation.from_config(simulation_config)
        self.regions = RegionMasks.from_size(simulation_config.field.size)
        self.renderer = ContinuousAudioRenderer(
            sample_rate=config.sample_rate,
            frame_size=config.frame_size,
            carrier_frequency=config.carrier_frequency,
            frequency_scale=config.frequency_scale,
            gain=config.gain,
            smoothing=config.smoothing,
        )
        self.callback_count = 0

    def callback(
        self,
        outdata: NDArray[np.float32],
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del time_info
        if status:
            print(status)

        for _ in range(self.config.physics_steps_per_audio_frame):
            frame = self.simulation.step()

        if frames != self.renderer.frame_size:
            self.renderer = ContinuousAudioRenderer(
                sample_rate=self.config.sample_rate,
                frame_size=frames,
                carrier_frequency=self.config.carrier_frequency,
                frequency_scale=self.config.frequency_scale,
                gain=self.config.gain,
                smoothing=self.config.smoothing,
            )

        audio = self.renderer.render_frame(frame.state, self.regions)
        outdata[:, 0] = np.asarray(audio, dtype=np.float32)
        self.callback_count += 1


def make_sounddevice_stream_factory() -> StreamFactory:
    sounddevice = cast(Any, importlib.import_module("sounddevice"))
    return cast(StreamFactory, sounddevice.OutputStream)


def play_realtime_audio(
    config: RealtimeAudioConfig,
    *,
    stream_factory: StreamFactory | None = None,
) -> RealtimeAudioEngine:
    engine = RealtimeAudioEngine(config)
    factory = stream_factory or make_sounddevice_stream_factory()
    stream = factory(
        samplerate=config.sample_rate,
        blocksize=config.frame_size,
        channels=1,
        dtype="float32",
        callback=engine.callback,
    )

    with stream:
        if config.duration_seconds is None:
            print("Realtime audio running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.25)
            except KeyboardInterrupt:
                pass
        else:
            time.sleep(config.duration_seconds)
    return engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play realtime audio generated from the output field region.",
    )
    parser.add_argument("--config", type=Path, default=RealtimeAudioConfig.config_path)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--gain", type=float, default=0.2)
    parser.add_argument("--carrier-frequency", type=float, default=220.0)
    parser.add_argument("--frequency-scale", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--physics-steps-per-audio-frame", type=int, default=1)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Optional finite playback duration. Omit to run until Ctrl+C.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RealtimeAudioConfig(
        config_path=args.config,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        gain=args.gain,
        carrier_frequency=args.carrier_frequency,
        frequency_scale=args.frequency_scale,
        smoothing=args.smoothing,
        physics_steps_per_audio_frame=args.physics_steps_per_audio_frame,
        duration_seconds=args.duration_seconds,
    )
    play_realtime_audio(config)
    return 0
