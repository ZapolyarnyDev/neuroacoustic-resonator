from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from neuroacoustic_resonator.analysis.metrics import RegionalActivityTracker
from neuroacoustic_resonator.analysis.output_patterns import OutputPatternHistory
from neuroacoustic_resonator.analysis.output_patterns import output_pattern_signature
from neuroacoustic_resonator.analysis.pattern_plasticity import (
    PatternGuidedPlasticityConfig,
    PatternPlasticityDecision,
    summarize_plasticity_decisions,
)
from neuroacoustic_resonator.audio.conversation import (
    drive_utterance,
    render_field_response,
    summarize_pattern_audio,
)
from neuroacoustic_resonator.audio.conversation_presets import (
    conversation_preset,
    preset_names,
)
from neuroacoustic_resonator.audio.input import (
    WavInputDrive,
    extract_audio_array_features,
)
from neuroacoustic_resonator.audio.output import (
    VoiceResponseSonificationRenderer,
    write_wav,
)
from neuroacoustic_resonator.audio.render import steps_for_duration
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation


class InputStreamLike(Protocol):
    def __enter__(self) -> InputStreamLike: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def read(self, frames: int) -> tuple[np.ndarray, bool]: ...


class SoundDeviceLike(Protocol):
    def play(
        self,
        data: np.ndarray,
        *,
        samplerate: int,
        blocking: bool,
        device: int | str | None = None,
    ) -> None: ...

    def InputStream(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        dtype: str,
        device: int | str | None = None,
    ) -> InputStreamLike: ...


@dataclass(frozen=True)
class LiveConversationConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    sample_rate: int = 48_000
    input_block_size: int = 1024
    output_frame_size: int = 512
    input_frame_size: int = 1024
    input_hop_size: int = 512
    drive_strength: float = 0.45
    input_assoc_gain: float = 0.8
    input_output_gain: float = 0.0
    response_seconds: float = 1.5
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
    warmup_steps: int = 100
    gain: float = 0.35
    carrier_frequency: float = 220.0
    frequency_scale: float = 1.0
    response_threshold: float = 0.0
    response_sensitivity: float = 900.0
    pattern_voice_depth: float = 0.55
    preset_name: str | None = None
    pattern_guided_plasticity: PatternGuidedPlasticityConfig = (
        PatternGuidedPlasticityConfig()
    )
    start_rms: float = 0.015
    stop_rms: float = 0.008
    min_utterance_seconds: float = 0.2
    max_utterance_seconds: float = 8.0
    silence_seconds: float = 0.45
    max_turns: int | None = None
    input_device: int | str | None = None
    output_device: int | str | None = None
    print_rms: bool = False
    print_pattern_telemetry: bool = True
    rms_report_interval: float = 0.5
    idle_timeout_seconds: float | None = None
    record_dir: Path | None = None
    summary_path: Path | None = Path("experiments") / "logs" / "live_conversation.json"

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if self.input_block_size < 1:
            msg = "input_block_size must be positive"
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
        if self.min_response_seconds <= 0.0:
            msg = "min_response_seconds must be positive"
            raise ValueError(msg)
        if self.max_response_seconds < self.min_response_seconds:
            msg = "max_response_seconds must be at least min_response_seconds"
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
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
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
        if self.response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)
        if self.response_sensitivity <= 0.0:
            msg = "response_sensitivity must be positive"
            raise ValueError(msg)
        if self.pattern_voice_depth < 0.0:
            msg = "pattern_voice_depth must be non-negative"
            raise ValueError(msg)
        if self.preset_name is not None:
            conversation_preset(self.preset_name)
        if self.start_rms < 0.0:
            msg = "start_rms must be non-negative"
            raise ValueError(msg)
        if self.stop_rms < 0.0:
            msg = "stop_rms must be non-negative"
            raise ValueError(msg)
        if self.min_utterance_seconds <= 0.0:
            msg = "min_utterance_seconds must be positive"
            raise ValueError(msg)
        if self.max_utterance_seconds < self.min_utterance_seconds:
            msg = "max_utterance_seconds must be at least min_utterance_seconds"
            raise ValueError(msg)
        if self.silence_seconds <= 0.0:
            msg = "silence_seconds must be positive"
            raise ValueError(msg)
        if self.max_turns is not None and self.max_turns < 1:
            msg = "max_turns must be positive when set"
            raise ValueError(msg)
        if self.rms_report_interval <= 0.0:
            msg = "rms_report_interval must be positive"
            raise ValueError(msg)
        if self.idle_timeout_seconds is not None and self.idle_timeout_seconds <= 0.0:
            msg = "idle_timeout_seconds must be positive when set"
            raise ValueError(msg)


@dataclass(frozen=True)
class LiveTurnResult:
    index: int
    input_audio: np.ndarray
    response_audio: np.ndarray
    summary: dict[str, Any]


class LiveConversationEngine:
    def __init__(self, config: LiveConversationConfig) -> None:
        self.config = apply_live_conversation_preset(config)
        config = self.config
        sim_config = SimulationConfig.from_file(config.config_path)
        self.simulation = Simulation.from_config(sim_config)
        self.regions = RegionMasks.from_size(sim_config.field.size)
        self.tracker = RegionalActivityTracker()
        self.renderer = VoiceResponseSonificationRenderer(
            sample_rate=config.sample_rate,
            frame_size=config.output_frame_size,
            carrier_frequency=config.carrier_frequency,
            frequency_scale=config.frequency_scale,
            gain=config.gain,
            response_threshold=config.response_threshold,
            response_sensitivity=config.response_sensitivity,
            pattern_voice_depth=config.pattern_voice_depth,
        )
        self.turns: list[dict[str, Any]] = []
        if config.record_dir is not None:
            config.record_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(config.warmup_steps):
            frame = self.simulation.step()
            self.tracker.update(
                frame,
                self.regions,
                input_value=self.simulation.last_input_value,
            )

    def process_utterance(self, audio: np.ndarray, *, index: int) -> LiveTurnResult:
        samples = np.asarray(audio, dtype=np.float64).reshape(-1)
        features = extract_audio_array_features(
            samples,
            sample_rate=self.config.sample_rate,
            frame_size=self.config.input_frame_size,
            hop_size=self.config.input_hop_size,
            drive_strength=self.config.drive_strength,
        )
        drive = WavInputDrive(
            features,
            self.regions,
            assoc_gain=self.config.input_assoc_gain,
            output_gain=self.config.input_output_gain,
        )
        drive_result = drive_utterance(
            self.simulation,
            self.tracker,
            self.regions,
            features,
            drive,
        )
        input_end_pattern = output_pattern_signature(
            self.simulation.field.state,
            self.regions,
        )
        planned_response_seconds = response_duration_for_live_input(
            drive_result,
            config=self.config,
        )
        response_steps = steps_for_duration(
            planned_response_seconds,
            sample_rate=self.config.sample_rate,
            frame_size=self.config.output_frame_size,
        )
        response_pattern_history = OutputPatternHistory()
        plasticity_decisions: list[PatternPlasticityDecision] = []
        response_audio, response_scores = render_field_response(
            self.simulation,
            self.tracker,
            self.regions,
            self.renderer,
            response_steps=response_steps,
            initial_response_score=drive_result.response_seed
            * self.config.response_seed_gain,
            seed_decay_seconds=self.config.response_seed_decay_seconds,
            sample_rate=self.config.sample_rate,
            output_plasticity_rate=self.config.output_plasticity_rate,
            output_frequency_plasticity_rate=self.config.output_frequency_plasticity_rate,
            pattern_history=response_pattern_history,
            pattern_guided_plasticity=self.config.pattern_guided_plasticity,
            plasticity_decisions=plasticity_decisions,
        )
        response_end_pattern = output_pattern_signature(
            self.simulation.field.state,
            self.regions,
        )
        response_audio_diagnostics = summarize_pattern_audio(
            response_audio,
            response_pattern_history,
            sample_rate=self.config.sample_rate,
            frame_size=self.config.output_frame_size,
        )
        summary: dict[str, Any] = {
            "index": index,
            "input_samples": int(samples.size),
            "input_duration_seconds": float(samples.size / self.config.sample_rate),
            "input_frames": features.frame_count,
            "peak_input_value": float(np.max(drive_result.input_values)),
            "mean_input_value": float(np.mean(drive_result.input_values)),
            "peak_input_fast_response_score": float(
                np.max(drive_result.fast_response_scores)
            ),
            "peak_input_event_score": float(np.max(drive_result.event_scores)),
            "initial_response_seed": float(
                drive_result.response_seed * self.config.response_seed_gain
            ),
            "planned_response_seconds": planned_response_seconds,
            "response_steps": response_steps,
            "response_duration_seconds": float(
                response_steps * self.config.output_frame_size / self.config.sample_rate
            ),
            "peak_response_score": float(np.max(response_scores)),
            "mean_response_score": float(np.mean(response_scores)),
            "input_output_pattern_history": drive_result.output_pattern_summary,
            "response_output_pattern_history": response_pattern_history.summary(),
            "response_pattern_audio_diagnostics": response_audio_diagnostics,
            "pattern_guided_plasticity": summarize_plasticity_decisions(
                plasticity_decisions
            ),
            "input_end_output_pattern": input_end_pattern.to_dict(),
            "response_end_output_pattern": response_end_pattern.to_dict(),
        }
        summary["pattern_telemetry"] = live_turn_pattern_telemetry(summary)
        if self.config.record_dir is not None:
            input_path = self.config.record_dir / f"turn_{index:03d}_input.wav"
            response_path = self.config.record_dir / f"turn_{index:03d}_response.wav"
            write_wav(input_path, samples, sample_rate=self.config.sample_rate)
            write_wav(
                response_path,
                response_audio,
                sample_rate=self.config.sample_rate,
            )
            summary["input_wav"] = str(input_path)
            summary["response_wav"] = str(response_path)
        self.turns.append(summary)
        return LiveTurnResult(
            index=index,
            input_audio=samples,
            response_audio=response_audio,
            summary=summary,
        )

    def session_summary(self) -> dict[str, Any]:
        return {
            "parameters": live_config_parameters(self.config),
            "turn_count": len(self.turns),
            "turns": self.turns,
        }


def response_duration_for_live_input(
    drive_result: Any,
    *,
    config: LiveConversationConfig,
) -> float:
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
        np.clip(duration, config.min_response_seconds, config.max_response_seconds)
    )


def apply_live_conversation_preset(
    config: LiveConversationConfig,
) -> LiveConversationConfig:
    if config.preset_name is None:
        return config
    preset = conversation_preset(config.preset_name)
    return replace(
        config,
        gain=preset.gain,
        response_seconds=preset.response_seconds,
        min_response_seconds=preset.min_response_seconds,
        max_response_seconds=preset.max_response_seconds,
        response_seed_gain=preset.response_seed_gain,
        response_seed_decay_seconds=preset.response_seed_decay_seconds,
        output_plasticity_rate=preset.output_plasticity_rate,
        output_frequency_plasticity_rate=preset.output_frequency_plasticity_rate,
        response_threshold=preset.response_threshold,
        response_sensitivity=preset.response_sensitivity,
        pattern_voice_depth=preset.pattern_voice_depth,
    )


def record_utterance(
    config: LiveConversationConfig,
    sounddevice: SoundDeviceLike,
) -> np.ndarray | None:
    blocks: list[np.ndarray] = []
    started = False
    silence_blocks = 0
    min_blocks = seconds_to_blocks(config.min_utterance_seconds, config)
    max_blocks = seconds_to_blocks(config.max_utterance_seconds, config)
    stop_blocks = seconds_to_blocks(config.silence_seconds, config)
    report_every_blocks = seconds_to_blocks(config.rms_report_interval, config)
    idle_blocks = (
        seconds_to_blocks(config.idle_timeout_seconds, config)
        if config.idle_timeout_seconds is not None
        else None
    )

    stream = sounddevice.InputStream(
        samplerate=config.sample_rate,
        blocksize=config.input_block_size,
        channels=1,
        dtype="float32",
        device=config.input_device,
    )
    block_index = 0
    with stream:
        while True:
            if started and len(blocks) >= max_blocks:
                break
            if not started and idle_blocks is not None and block_index >= idle_blocks:
                return None
            block, overflowed = stream.read(config.input_block_size)
            mono = np.asarray(block, dtype=np.float64).reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if config.print_rms and block_index % report_every_blocks == 0:
                overflow = " overflow" if overflowed else ""
                print(
                    f"mic rms={rms:.5f} "
                    f"start={config.start_rms:.5f} stop={config.stop_rms:.5f}"
                    f"{overflow}"
                )
            block_index += 1

            if not started:
                if rms < config.start_rms:
                    continue
                started = True

            blocks.append(mono)
            if rms <= config.stop_rms:
                silence_blocks += 1
            else:
                silence_blocks = 0
            if len(blocks) >= min_blocks and silence_blocks >= stop_blocks:
                break

    if not blocks:
        return None
    audio = np.concatenate(blocks)
    return trim_tail_silence(audio, config=config)


def trim_tail_silence(
    audio: np.ndarray, *, config: LiveConversationConfig
) -> np.ndarray:
    block_size = config.input_block_size
    stop_blocks = seconds_to_blocks(config.silence_seconds, config)
    trim_samples = stop_blocks * block_size
    if audio.size <= trim_samples:
        return audio
    return audio[: audio.size - trim_samples]


def seconds_to_blocks(seconds: float, config: LiveConversationConfig) -> int:
    block_seconds = config.input_block_size / config.sample_rate
    return max(1, int(np.ceil(seconds / block_seconds)))


def live_config_parameters(config: LiveConversationConfig) -> dict[str, Any]:
    return {
        "config_path": str(config.config_path),
        "sample_rate": config.sample_rate,
        "input_block_size": config.input_block_size,
        "output_frame_size": config.output_frame_size,
        "input_frame_size": config.input_frame_size,
        "input_hop_size": config.input_hop_size,
        "drive_strength": config.drive_strength,
        "input_assoc_gain": config.input_assoc_gain,
        "input_output_gain": config.input_output_gain,
        "response_seconds": config.response_seconds,
        "min_response_seconds": config.min_response_seconds,
        "max_response_seconds": config.max_response_seconds,
        "output_plasticity_rate": config.output_plasticity_rate,
        "output_frequency_plasticity_rate": config.output_frequency_plasticity_rate,
        "preset_name": config.preset_name,
        "pattern_voice_depth": config.pattern_voice_depth,
        "pattern_guided_plasticity": config.pattern_guided_plasticity.enabled,
        "pattern_guided_output_gain": config.pattern_guided_plasticity.output_gain,
        "pattern_guided_assoc_gain": config.pattern_guided_plasticity.assoc_gain,
        "start_rms": config.start_rms,
        "stop_rms": config.stop_rms,
        "silence_seconds": config.silence_seconds,
        "max_utterance_seconds": config.max_utterance_seconds,
        "max_turns": config.max_turns,
        "input_device": config.input_device,
        "output_device": config.output_device,
        "print_rms": config.print_rms,
        "print_pattern_telemetry": config.print_pattern_telemetry,
        "idle_timeout_seconds": config.idle_timeout_seconds,
        "record_dir": str(config.record_dir) if config.record_dir is not None else None,
    }


def write_live_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def live_turn_pattern_telemetry(summary: dict[str, Any]) -> dict[str, Any]:
    response_history = summary["response_output_pattern_history"]
    diagnostics = summary["response_pattern_audio_diagnostics"]["overall"]
    return {
        "active_pattern_label": response_history["active_dominant_label"],
        "dominant_pattern_label": response_history["dominant_label"],
        "pattern_confidence": response_history["mean_confidence"],
        "peak_activation": response_history["peak_activation"],
        "peak_activation_label": response_history["peak_activation_label"],
        "peak_response_score": summary["peak_response_score"],
        "response_audio_rms": diagnostics["rms"],
        "response_audio_spectral_centroid_hz": diagnostics["spectral_centroid_hz"],
        "response_duration_seconds": summary["response_duration_seconds"],
    }


def format_live_pattern_telemetry(telemetry: dict[str, Any]) -> str:
    label = telemetry["active_pattern_label"] or telemetry["dominant_pattern_label"]
    if label is None:
        label = "none"
    return (
        "pattern="
        f"{label} "
        f"confidence={telemetry['pattern_confidence']:.3f} "
        f"peak={telemetry['peak_response_score']:.6f} "
        f"rms={telemetry['response_audio_rms']:.4f} "
        f"centroid={telemetry['response_audio_spectral_centroid_hz']:.1f}Hz"
    )


def make_sounddevice() -> SoundDeviceLike:
    return cast(SoundDeviceLike, importlib.import_module("sounddevice"))


def print_audio_devices() -> None:
    sounddevice = importlib.import_module("sounddevice")
    print(sounddevice.query_devices())


def run_live_conversation(
    config: LiveConversationConfig,
    *,
    sounddevice: SoundDeviceLike | None = None,
) -> LiveConversationEngine:
    sd = sounddevice or make_sounddevice()
    engine = LiveConversationEngine(config)
    print("Live conversation running. Speak, then pause. Press Ctrl+C to stop.")
    turn_index = 1
    try:
        while config.max_turns is None or turn_index <= config.max_turns:
            print(f"Listening for turn {turn_index}...")
            utterance = record_utterance(config, sd)
            if utterance is None:
                if config.idle_timeout_seconds is not None:
                    print(
                        "No utterance detected before idle timeout. "
                        "Try --print-rms, lower thresholds, or another input device."
                    )
                    break
                continue
            print(
                f"Processing turn {turn_index}: "
                f"{utterance.size / config.sample_rate:.2f}s"
            )
            result = engine.process_utterance(utterance, index=turn_index)
            sd.play(
                np.asarray(result.response_audio, dtype=np.float32),
                samplerate=config.sample_rate,
                blocking=True,
                device=config.output_device,
            )
            print(
                "Responded: "
                f"{result.summary['response_duration_seconds']:.2f}s "
                f"seed={result.summary['initial_response_seed']:.6f}"
            )
            if config.print_pattern_telemetry:
                print(
                    format_live_pattern_telemetry(result.summary["pattern_telemetry"])
                )
            turn_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if config.summary_path is not None:
            write_live_summary(config.summary_path, engine.session_summary())
            print(f"Wrote live conversation summary: {config.summary_path}")
        time.sleep(0.05)
    return engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a turn-based live microphone conversation with the field.",
    )
    parser.add_argument(
        "--config", type=Path, default=LiveConversationConfig.config_path
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--input-block-size", type=int, default=1024)
    parser.add_argument("--output-frame-size", type=int, default=512)
    parser.add_argument("--input-frame-size", type=int, default=1024)
    parser.add_argument("--input-hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--input-assoc-gain", type=float, default=0.8)
    parser.add_argument("--input-output-gain", type=float, default=0.0)
    parser.add_argument("--response-seconds", type=float, default=1.5)
    parser.add_argument("--min-response-seconds", type=float, default=0.6)
    parser.add_argument("--max-response-seconds", type=float, default=3.0)
    parser.add_argument("--output-plasticity-rate", type=float, default=0.02)
    parser.add_argument("--output-frequency-plasticity-rate", type=float, default=0.004)
    parser.add_argument("--gain", type=float, default=0.35)
    parser.add_argument("--preset", choices=preset_names(), default=None)
    parser.add_argument("--pattern-voice-depth", type=float, default=0.55)
    parser.add_argument("--start-rms", type=float, default=0.015)
    parser.add_argument("--stop-rms", type=float, default=0.008)
    parser.add_argument("--silence-seconds", type=float, default=0.45)
    parser.add_argument("--max-utterance-seconds", type=float, default=8.0)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--print-rms", action="store_true")
    parser.add_argument(
        "--no-pattern-telemetry",
        action="store_true",
        help="Do not print per-turn active pattern telemetry.",
    )
    parser.add_argument("--rms-report-interval", type=float, default=0.5)
    parser.add_argument("--idle-timeout-seconds", type=float, default=None)
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice input/output devices and exit.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=LiveConversationConfig.summary_path,
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write a JSON session summary on exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        print_audio_devices()
        return 0
    try:
        run_live_conversation(
            LiveConversationConfig(
                config_path=args.config,
                sample_rate=args.sample_rate,
                input_block_size=args.input_block_size,
                output_frame_size=args.output_frame_size,
                input_frame_size=args.input_frame_size,
                input_hop_size=args.input_hop_size,
                drive_strength=args.drive_strength,
                input_assoc_gain=args.input_assoc_gain,
                input_output_gain=args.input_output_gain,
                response_seconds=args.response_seconds,
                min_response_seconds=args.min_response_seconds,
                max_response_seconds=args.max_response_seconds,
                output_plasticity_rate=args.output_plasticity_rate,
                output_frequency_plasticity_rate=args.output_frequency_plasticity_rate,
                gain=args.gain,
                preset_name=args.preset,
                pattern_voice_depth=args.pattern_voice_depth,
                start_rms=args.start_rms,
                stop_rms=args.stop_rms,
                silence_seconds=args.silence_seconds,
                max_utterance_seconds=args.max_utterance_seconds,
                max_turns=args.max_turns,
                input_device=parse_device(args.input_device),
                output_device=parse_device(args.output_device),
                print_rms=args.print_rms,
                print_pattern_telemetry=not args.no_pattern_telemetry,
                rms_report_interval=args.rms_report_interval,
                idle_timeout_seconds=args.idle_timeout_seconds,
                record_dir=args.record_dir,
                summary_path=None if args.no_summary else args.summary,
            )
        )
    except Exception as exc:
        print(f"Live conversation audio error: {exc}")
        return 2
    return 0


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return stripped
