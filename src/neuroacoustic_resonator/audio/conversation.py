from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.metrics import RegionalActivityTracker
from neuroacoustic_resonator.audio.input import (
    WavInputDrive,
    extract_audio_input_features,
)
from neuroacoustic_resonator.audio.output import (
    VoiceResponseSonificationRenderer,
    write_wav,
)
from neuroacoustic_resonator.audio.render import steps_for_duration
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame

ConversationSummary = dict[str, Any]


@dataclass(frozen=True)
class UtteranceDriveResult:
    input_values: np.ndarray
    fast_response_scores: np.ndarray
    event_scores: np.ndarray

    @property
    def response_seed(self) -> float:
        peak_fast = (
            float(np.max(self.fast_response_scores))
            if self.fast_response_scores.size
            else 0.0
        )
        peak_event = float(np.max(self.event_scores)) if self.event_scores.size else 0.0
        return max(peak_fast, peak_event)


@dataclass(frozen=True)
class VoiceConversationConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    input_wavs: tuple[Path, ...] = ()
    output_wav: Path = Path("experiments") / "audio" / "voice-conversation.wav"
    output_summary: Path = (
        Path("experiments") / "logs" / "voice_conversation_summary.json"
    )
    sample_rate: int = 48_000
    output_frame_size: int = 512
    input_frame_size: int = 1024
    input_hop_size: int = 512
    drive_strength: float = 0.45
    input_assoc_gain: float = 0.8
    input_output_gain: float = 0.0
    response_seconds: float = 1.5
    pause_seconds: float = 0.25
    warmup_steps: int = 100
    gain: float = 0.35
    input_mix_gain: float = 0.8
    include_input_audio: bool = True
    use_response_policy: bool = True
    min_response_seconds: float = 0.6
    max_response_seconds: float = 3.0
    input_peak_response_gain: float = 3.0
    input_mean_response_gain: float = 1.0
    fast_response_duration_gain: float = 120.0
    event_response_duration_gain: float = 80.0
    response_seed_gain: float = 0.65
    response_seed_decay_seconds: float = 0.7
    output_plasticity_rate: float = 0.02
    output_frequency_plasticity_rate: float = 0.004
    carrier_frequency: float = 220.0
    frequency_scale: float = 1.0
    response_threshold: float = 0.0
    response_sensitivity: float = 900.0

    def __post_init__(self) -> None:
        if not self.input_wavs:
            msg = "input_wavs must not be empty"
            raise ValueError(msg)
        if self.sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if self.output_frame_size < 1:
            msg = "output_frame_size must be positive"
            raise ValueError(msg)
        if self.input_frame_size < 1:
            msg = "input_frame_size must be positive"
            raise ValueError(msg)
        if self.input_hop_size < 1:
            msg = "input_hop_size must be positive"
            raise ValueError(msg)
        if self.drive_strength < 0.0:
            msg = "drive_strength must be non-negative"
            raise ValueError(msg)
        if self.input_assoc_gain < 0.0:
            msg = "input_assoc_gain must be non-negative"
            raise ValueError(msg)
        if self.input_output_gain < 0.0:
            msg = "input_output_gain must be non-negative"
            raise ValueError(msg)
        if self.response_seconds <= 0.0:
            msg = "response_seconds must be positive"
            raise ValueError(msg)
        if self.pause_seconds < 0.0:
            msg = "pause_seconds must be non-negative"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.gain < 0.0:
            msg = "gain must be non-negative"
            raise ValueError(msg)
        if self.input_mix_gain < 0.0:
            msg = "input_mix_gain must be non-negative"
            raise ValueError(msg)
        if self.min_response_seconds <= 0.0:
            msg = "min_response_seconds must be positive"
            raise ValueError(msg)
        if self.max_response_seconds < self.min_response_seconds:
            msg = "max_response_seconds must be at least min_response_seconds"
            raise ValueError(msg)
        if self.input_peak_response_gain < 0.0:
            msg = "input_peak_response_gain must be non-negative"
            raise ValueError(msg)
        if self.input_mean_response_gain < 0.0:
            msg = "input_mean_response_gain must be non-negative"
            raise ValueError(msg)
        if self.fast_response_duration_gain < 0.0:
            msg = "fast_response_duration_gain must be non-negative"
            raise ValueError(msg)
        if self.event_response_duration_gain < 0.0:
            msg = "event_response_duration_gain must be non-negative"
            raise ValueError(msg)
        if self.response_seed_gain < 0.0:
            msg = "response_seed_gain must be non-negative"
            raise ValueError(msg)
        if self.response_seed_decay_seconds <= 0.0:
            msg = "response_seed_decay_seconds must be positive"
            raise ValueError(msg)
        if self.output_plasticity_rate < 0.0:
            msg = "output_plasticity_rate must be non-negative"
            raise ValueError(msg)
        if self.output_frequency_plasticity_rate < 0.0:
            msg = "output_frequency_plasticity_rate must be non-negative"
            raise ValueError(msg)
        if self.carrier_frequency <= 0.0:
            msg = "carrier_frequency must be positive"
            raise ValueError(msg)
        if self.frequency_scale <= 0.0:
            msg = "frequency_scale must be positive"
            raise ValueError(msg)
        if self.response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)
        if self.response_sensitivity <= 0.0:
            msg = "response_sensitivity must be positive"
            raise ValueError(msg)


def render_voice_conversation(config: VoiceConversationConfig) -> ConversationSummary:
    sim_config = SimulationConfig.from_file(config.config_path)
    simulation = Simulation.from_config(sim_config)
    regions = RegionMasks.from_size(sim_config.field.size)
    tracker = RegionalActivityTracker()
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=config.sample_rate,
        frame_size=config.output_frame_size,
        carrier_frequency=config.carrier_frequency,
        frequency_scale=config.frequency_scale,
        gain=config.gain,
        response_threshold=config.response_threshold,
        response_sensitivity=config.response_sensitivity,
    )

    for _ in range(config.warmup_steps):
        frame = simulation.step()
        tracker.update(frame, regions, input_value=simulation.last_input_value)

    pause_samples = int(round(config.pause_seconds * config.sample_rate))
    audio_frames: list[np.ndarray] = []
    utterances: list[dict[str, Any]] = []

    for utterance_index, input_wav in enumerate(config.input_wavs, start=1):
        input_audio = (
            read_conversation_input_audio(
                input_wav,
                sample_rate=config.sample_rate,
                gain=config.input_mix_gain,
            )
            if config.include_input_audio
            else np.zeros(0, dtype=np.float64)
        )
        features = extract_audio_input_features(
            input_wav,
            frame_size=config.input_frame_size,
            hop_size=config.input_hop_size,
            drive_strength=config.drive_strength,
        )
        drive = WavInputDrive(
            features,
            regions,
            assoc_gain=config.input_assoc_gain,
            output_gain=config.input_output_gain,
        )
        drive_result = drive_utterance(simulation, tracker, regions, features, drive)
        planned_response_seconds = response_duration_for_input(
            drive_result,
            config=config,
        )
        response_steps = steps_for_duration(
            planned_response_seconds,
            sample_rate=config.sample_rate,
            frame_size=config.output_frame_size,
        )
        response_audio, response_scores = render_field_response(
            simulation,
            tracker,
            regions,
            renderer,
            response_steps=response_steps,
            initial_response_score=drive_result.response_seed
            * config.response_seed_gain,
            seed_decay_seconds=config.response_seed_decay_seconds,
            sample_rate=config.sample_rate,
            output_plasticity_rate=config.output_plasticity_rate,
            output_frequency_plasticity_rate=config.output_frequency_plasticity_rate,
        )
        if input_audio.size:
            audio_frames.append(input_audio)
            if pause_samples:
                audio_frames.append(np.zeros(pause_samples, dtype=np.float64))
        audio_frames.append(response_audio)
        if pause_samples:
            audio_frames.append(np.zeros(pause_samples, dtype=np.float64))
        utterances.append(
            {
                "index": utterance_index,
                "input_wav": str(input_wav),
                "input_frames": features.frame_count,
                "input_duration_seconds": features.duration_seconds,
                "mixed_input_audio_seconds": float(
                    input_audio.size / config.sample_rate
                ),
                "peak_input_value": float(np.max(drive_result.input_values)),
                "mean_input_value": float(np.mean(drive_result.input_values)),
                "peak_input_fast_response_score": float(
                    np.max(drive_result.fast_response_scores)
                ),
                "mean_input_fast_response_score": float(
                    np.mean(drive_result.fast_response_scores)
                ),
                "peak_input_event_score": float(np.max(drive_result.event_scores)),
                "mean_input_event_score": float(np.mean(drive_result.event_scores)),
                "initial_response_seed": float(
                    drive_result.response_seed * config.response_seed_gain
                ),
                "planned_response_seconds": planned_response_seconds,
                "response_steps": response_steps,
                "response_duration_seconds": (
                    response_steps * config.output_frame_size / config.sample_rate
                ),
                "peak_response_score": float(np.max(response_scores)),
                "mean_response_score": float(np.mean(response_scores)),
            }
        )

    audio = (
        np.concatenate(audio_frames) if audio_frames else np.zeros(0, dtype=np.float64)
    )
    write_wav(config.output_wav, audio, sample_rate=config.sample_rate)
    summary: ConversationSummary = {
        "config": str(config.config_path),
        "output_wav": str(config.output_wav),
        "parameters": {
            "sample_rate": config.sample_rate,
            "output_frame_size": config.output_frame_size,
            "input_frame_size": config.input_frame_size,
            "input_hop_size": config.input_hop_size,
            "drive_strength": config.drive_strength,
            "input_assoc_gain": config.input_assoc_gain,
            "input_output_gain": config.input_output_gain,
            "response_seconds": config.response_seconds,
            "pause_seconds": config.pause_seconds,
            "warmup_steps": config.warmup_steps,
            "gain": config.gain,
            "input_mix_gain": config.input_mix_gain,
            "include_input_audio": config.include_input_audio,
            "use_response_policy": config.use_response_policy,
            "min_response_seconds": config.min_response_seconds,
            "max_response_seconds": config.max_response_seconds,
            "input_peak_response_gain": config.input_peak_response_gain,
            "input_mean_response_gain": config.input_mean_response_gain,
            "fast_response_duration_gain": config.fast_response_duration_gain,
            "event_response_duration_gain": config.event_response_duration_gain,
            "response_seed_gain": config.response_seed_gain,
            "response_seed_decay_seconds": config.response_seed_decay_seconds,
            "output_plasticity_rate": config.output_plasticity_rate,
            "output_frequency_plasticity_rate": config.output_frequency_plasticity_rate,
            "carrier_frequency": config.carrier_frequency,
            "frequency_scale": config.frequency_scale,
            "response_threshold": config.response_threshold,
            "response_sensitivity": config.response_sensitivity,
        },
        "utterance_count": len(utterances),
        "duration_seconds": float(audio.size / config.sample_rate),
        "session": summarize_conversation_session(
            utterances,
            output_duration_seconds=float(audio.size / config.sample_rate),
        ),
        "utterances": utterances,
    }
    write_conversation_summary(config.output_summary, summary)
    return summary


def summarize_conversation_session(
    utterances: list[dict[str, Any]],
    *,
    output_duration_seconds: float,
) -> dict[str, float | int | None]:
    if not utterances:
        return {
            "utterance_count": 0,
            "output_duration_seconds": output_duration_seconds,
            "total_input_audio_seconds": 0.0,
            "total_response_audio_seconds": 0.0,
            "peak_input_value": 0.0,
            "peak_response_score": 0.0,
            "mean_peak_response_score": 0.0,
            "mean_planned_response_seconds": 0.0,
            "strongest_input_index": None,
            "strongest_response_index": None,
        }

    peak_inputs = np.asarray(
        [utterance["peak_input_value"] for utterance in utterances],
        dtype=np.float64,
    )
    peak_responses = np.asarray(
        [utterance["peak_response_score"] for utterance in utterances],
        dtype=np.float64,
    )
    planned_response_seconds = np.asarray(
        [utterance["planned_response_seconds"] for utterance in utterances],
        dtype=np.float64,
    )
    return {
        "utterance_count": len(utterances),
        "output_duration_seconds": output_duration_seconds,
        "total_input_audio_seconds": float(
            sum(utterance["mixed_input_audio_seconds"] for utterance in utterances)
        ),
        "total_response_audio_seconds": float(
            sum(utterance["response_duration_seconds"] for utterance in utterances)
        ),
        "peak_input_value": float(np.max(peak_inputs)),
        "peak_response_score": float(np.max(peak_responses)),
        "mean_peak_response_score": float(np.mean(peak_responses)),
        "mean_planned_response_seconds": float(np.mean(planned_response_seconds)),
        "strongest_input_index": int(utterances[int(np.argmax(peak_inputs))]["index"]),
        "strongest_response_index": int(
            utterances[int(np.argmax(peak_responses))]["index"]
        ),
    }


def response_duration_for_input(
    drive_result: UtteranceDriveResult | np.ndarray,
    *,
    config: VoiceConversationConfig,
) -> float:
    if not config.use_response_policy:
        return config.response_seconds
    if isinstance(drive_result, np.ndarray):
        input_scores = drive_result
        fast_scores = np.zeros_like(input_scores)
        event_scores = np.zeros_like(input_scores)
    else:
        input_scores = drive_result.input_values
        fast_scores = drive_result.fast_response_scores
        event_scores = drive_result.event_scores
    peak = float(np.max(input_scores)) if input_scores.size else 0.0
    mean = float(np.mean(input_scores)) if input_scores.size else 0.0
    mean_fast = float(np.mean(fast_scores)) if fast_scores.size else 0.0
    mean_event = float(np.mean(event_scores)) if event_scores.size else 0.0
    duration = (
        config.response_seconds
        + peak * config.input_peak_response_gain
        + mean * config.input_mean_response_gain
        + mean_fast * config.fast_response_duration_gain
        + mean_event * config.event_response_duration_gain
    )
    return float(
        np.clip(
            duration,
            config.min_response_seconds,
            config.max_response_seconds,
        )
    )


def read_conversation_input_audio(
    path: str | Path,
    *,
    sample_rate: int,
    gain: float,
) -> np.ndarray:
    source_rate, samples = wavfile.read(path)
    audio = to_mono_float(samples)
    if int(source_rate) != sample_rate:
        common = math.gcd(int(source_rate), sample_rate)
        audio = resample_poly(audio, sample_rate // common, int(source_rate) // common)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float64, copy=False)


def to_mono_float(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        msg = "WAV input must be mono or stereo"
        raise ValueError(msg)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        integer_peak = max(abs(info.min), abs(info.max))
        return audio.astype(np.float64) / integer_peak
    floating = audio.astype(np.float64)
    floating_peak = float(np.max(np.abs(floating))) if floating.size else 0.0
    if floating_peak > 1.0:
        floating = floating / floating_peak
    return floating


def drive_utterance(
    simulation: Simulation,
    tracker: RegionalActivityTracker,
    regions: RegionMasks,
    features: Any,
    drive: WavInputDrive,
) -> UtteranceDriveResult:
    input_values: list[float] = []
    fast_response_scores: list[float] = []
    event_scores: list[float] = []
    for input_step in range(features.frame_count):
        input_value = drive.apply(simulation.field, input_step)
        simulation.step_index += 1
        state = simulation.field.step()
        simulation.last_input_value = input_value
        frame = SimulationFrame(
            state=state,
            metrics=simulation.field.metrics(step=simulation.step_index),
            local_synchrony=simulation.field.local_synchrony(),
        )
        metrics = tracker.update(frame, regions, input_value=input_value)
        input_values.append(input_value)
        fast_response_scores.append(metrics.output_fast_response_score)
        event_scores.append(metrics.output_event_score)
    return UtteranceDriveResult(
        input_values=np.asarray(input_values, dtype=np.float64),
        fast_response_scores=np.asarray(fast_response_scores, dtype=np.float64),
        event_scores=np.asarray(event_scores, dtype=np.float64),
    )


def render_field_response(
    simulation: Simulation,
    tracker: RegionalActivityTracker,
    regions: RegionMasks,
    renderer: VoiceResponseSonificationRenderer,
    *,
    response_steps: int,
    initial_response_score: float = 0.0,
    seed_decay_seconds: float = 0.7,
    sample_rate: int = 48_000,
    output_plasticity_rate: float = 0.0,
    output_frequency_plasticity_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    response_scores: list[float] = []
    decay_steps = max(1.0, seed_decay_seconds * sample_rate / renderer.frame_size)
    for response_step in range(response_steps):
        frame = simulation.step()
        metrics = tracker.update(
            frame, regions, input_value=simulation.last_input_value
        )
        response_score = max(
            metrics.output_fast_response_score,
            metrics.output_event_score,
            max(0.0, metrics.output_response_activity) * 0.05,
            initial_response_score * math.exp(-response_step / decay_steps),
        )
        simulation.field.apply_region_plasticity(
            regions.output,
            response_score,
            coupling_rate=output_plasticity_rate,
            frequency_rate=output_frequency_plasticity_rate,
        )
        frames.append(
            renderer.render_frame(
                frame.state,
                regions,
                response_score=response_score,
            )
        )
        response_scores.append(response_score)
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float64)
    scores = np.asarray(response_scores, dtype=np.float64)
    return audio, scores


def write_conversation_summary(
    path: str | Path,
    summary: ConversationSummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an offline field-response conversation from voice WAVs.",
    )
    parser.add_argument(
        "--config", type=Path, default=VoiceConversationConfig.config_path
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more voice WAV utterances.",
    )
    parser.add_argument(
        "--output", type=Path, default=VoiceConversationConfig.output_wav
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=VoiceConversationConfig.output_summary,
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--output-frame-size", type=int, default=512)
    parser.add_argument("--input-frame-size", type=int, default=1024)
    parser.add_argument("--input-hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--input-assoc-gain", type=float, default=0.8)
    parser.add_argument("--input-output-gain", type=float, default=0.0)
    parser.add_argument("--response-seconds", type=float, default=1.5)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gain", type=float, default=0.35)
    parser.add_argument("--input-mix-gain", type=float, default=0.8)
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="Write only generated field responses, without original input audio.",
    )
    parser.add_argument(
        "--fixed-response-duration",
        action="store_true",
        help="Disable input-strength response duration policy.",
    )
    parser.add_argument("--min-response-seconds", type=float, default=0.6)
    parser.add_argument("--max-response-seconds", type=float, default=3.0)
    parser.add_argument("--input-peak-response-gain", type=float, default=3.0)
    parser.add_argument("--input-mean-response-gain", type=float, default=1.0)
    parser.add_argument("--fast-response-duration-gain", type=float, default=120.0)
    parser.add_argument("--event-response-duration-gain", type=float, default=80.0)
    parser.add_argument("--response-seed-gain", type=float, default=0.65)
    parser.add_argument("--response-seed-decay-seconds", type=float, default=0.7)
    parser.add_argument("--output-plasticity-rate", type=float, default=0.02)
    parser.add_argument("--output-frequency-plasticity-rate", type=float, default=0.004)
    parser.add_argument("--carrier-frequency", type=float, default=220.0)
    parser.add_argument("--frequency-scale", type=float, default=1.0)
    parser.add_argument("--response-threshold", type=float, default=0.0)
    parser.add_argument("--response-sensitivity", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = render_voice_conversation(
        VoiceConversationConfig(
            config_path=args.config,
            input_wavs=tuple(args.inputs),
            output_wav=args.output,
            output_summary=args.summary,
            sample_rate=args.sample_rate,
            output_frame_size=args.output_frame_size,
            input_frame_size=args.input_frame_size,
            input_hop_size=args.input_hop_size,
            drive_strength=args.drive_strength,
            input_assoc_gain=args.input_assoc_gain,
            input_output_gain=args.input_output_gain,
            response_seconds=args.response_seconds,
            pause_seconds=args.pause_seconds,
            warmup_steps=args.warmup_steps,
            gain=args.gain,
            input_mix_gain=args.input_mix_gain,
            include_input_audio=not args.response_only,
            use_response_policy=not args.fixed_response_duration,
            min_response_seconds=args.min_response_seconds,
            max_response_seconds=args.max_response_seconds,
            input_peak_response_gain=args.input_peak_response_gain,
            input_mean_response_gain=args.input_mean_response_gain,
            fast_response_duration_gain=args.fast_response_duration_gain,
            event_response_duration_gain=args.event_response_duration_gain,
            response_seed_gain=args.response_seed_gain,
            response_seed_decay_seconds=args.response_seed_decay_seconds,
            output_plasticity_rate=args.output_plasticity_rate,
            output_frequency_plasticity_rate=args.output_frequency_plasticity_rate,
            carrier_frequency=args.carrier_frequency,
            frequency_scale=args.frequency_scale,
            response_threshold=args.response_threshold,
            response_sensitivity=args.response_sensitivity,
        )
    )
    print(
        "Rendered voice conversation: "
        f"{summary['output_wav']} "
        f"utterances={summary['utterance_count']} "
        f"duration={summary['duration_seconds']:.3f}s"
    )
    print(f"Wrote summary: {args.summary}")
    return 0
