from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.core.field import FieldState
from neuroacoustic_resonator.core.regions import RegionMasks

AudioArray = NDArray[np.float64]

TAU = 2.0 * np.pi


def render_output_frame(
    state: FieldState,
    regions: RegionMasks,
    *,
    sample_rate: int = 48_000,
    frame_size: int = 512,
    gain: float = 0.2,
) -> AudioArray:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)
    if gain < 0.0:
        msg = "gain must be non-negative"
        raise ValueError(msg)
    if state.phase.shape != regions.shape:
        msg = "state and regions must have matching shapes"
        raise ValueError(msg)

    mask = regions.output
    phase = state.phase[mask]
    frequency = np.clip(state.frequency[mask], 0.0, None)
    metabolite = state.metabolite[mask]
    coupling = state.coupling[mask]

    if phase.size == 0:
        return np.zeros(frame_size, dtype=np.float64)

    weights = np.clip(metabolite * (1.0 + coupling), 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones_like(weights) / float(weights.size)
    else:
        weights = weights / weight_sum

    time = np.arange(frame_size, dtype=np.float64) / float(sample_rate)
    angular_frequency = TAU * frequency[:, np.newaxis]
    oscillator_bank = np.sin(phase[:, np.newaxis] + angular_frequency * time)
    frame = gain * np.sum(weights[:, np.newaxis] * oscillator_bank, axis=0)

    return np.clip(frame, -1.0, 1.0).astype(np.float64, copy=False)


class ContinuousAudioRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
    ) -> None:
        if sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if frame_size < 1:
            msg = "frame_size must be positive"
            raise ValueError(msg)
        if carrier_frequency <= 0.0:
            msg = "carrier_frequency must be positive"
            raise ValueError(msg)
        if frequency_scale <= 0.0:
            msg = "frequency_scale must be positive"
            raise ValueError(msg)
        if gain < 0.0:
            msg = "gain must be non-negative"
            raise ValueError(msg)
        if not 0.0 < smoothing <= 1.0:
            msg = "smoothing must be in (0, 1]"
            raise ValueError(msg)

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.carrier_frequency = carrier_frequency
        self.frequency_scale = frequency_scale
        self.gain = gain
        self.smoothing = smoothing
        self._phase: AudioArray | None = None
        self._weights: AudioArray | None = None
        self._frequencies: AudioArray | None = None

    def render_frame(self, state: FieldState, regions: RegionMasks) -> AudioArray:
        if state.phase.shape != regions.shape:
            msg = "state and regions must have matching shapes"
            raise ValueError(msg)

        mask = regions.output
        field_frequency = np.clip(state.frequency[mask], 0.0, None)
        metabolite = state.metabolite[mask]
        coupling = state.coupling[mask]
        oscillator_count = field_frequency.size
        if oscillator_count == 0:
            return np.zeros(self.frame_size, dtype=np.float64)

        if self._phase is None or self._phase.size != oscillator_count:
            self._phase = np.mod(state.phase[mask], TAU).astype(np.float64, copy=True)
            self._weights = np.full(oscillator_count, 1.0 / oscillator_count)
            self._frequencies = np.full(oscillator_count, self.carrier_frequency)

        target_weights = self._normalized_weights(metabolite, coupling)
        target_frequencies = np.clip(
            self.carrier_frequency * self.frequency_scale * field_frequency,
            20.0,
            self.sample_rate / 2.0 - 1.0,
        )
        assert self._weights is not None
        assert self._frequencies is not None
        assert self._phase is not None
        self._weights = self._smooth(self._weights, target_weights)
        self._frequencies = self._smooth(self._frequencies, target_frequencies)

        samples = np.arange(self.frame_size, dtype=np.float64)
        increments = TAU * self._frequencies / float(self.sample_rate)
        phases = self._phase[:, np.newaxis] + increments[:, np.newaxis] * samples
        frame = self.gain * np.sum(
            self._weights[:, np.newaxis] * np.sin(phases),
            axis=0,
        )
        self._phase = np.mod(self._phase + increments * self.frame_size, TAU)
        return np.clip(frame, -1.0, 1.0).astype(np.float64, copy=False)

    def _smooth(self, current: AudioArray, target: AudioArray) -> AudioArray:
        return current + self.smoothing * (target - current)

    @staticmethod
    def _normalized_weights(metabolite: AudioArray, coupling: AudioArray) -> AudioArray:
        weights = np.clip(metabolite * (1.0 + coupling), 0.0, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return np.ones_like(weights) / float(weights.size)
        return weights / weight_sum


class GatedAudioRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        gate_threshold: float = 0.002,
        gate_sensitivity: float = 24.0,
        attack: float = 0.35,
        release: float = 0.04,
        baseline_smoothing: float = 0.01,
    ) -> None:
        if gate_threshold < 0.0:
            msg = "gate_threshold must be non-negative"
            raise ValueError(msg)
        if gate_sensitivity <= 0.0:
            msg = "gate_sensitivity must be positive"
            raise ValueError(msg)
        if not 0.0 < attack <= 1.0:
            msg = "attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < release <= 1.0:
            msg = "release must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < baseline_smoothing <= 1.0:
            msg = "baseline_smoothing must be in (0, 1]"
            raise ValueError(msg)

        self.continuous = ContinuousAudioRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.gate_threshold = gate_threshold
        self.gate_sensitivity = gate_sensitivity
        self.attack = attack
        self.release = release
        self.baseline_smoothing = baseline_smoothing
        self._baseline: float | None = None
        self.envelope = 0.0
        self.last_activation = 0.0

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    def render_frame(self, state: FieldState, regions: RegionMasks) -> AudioArray:
        signal = self._output_activity_signal(state, regions)
        if self._baseline is None:
            self._baseline = signal

        assert self._baseline is not None
        delta = abs(signal - self._baseline)
        self._baseline += self.baseline_smoothing * (signal - self._baseline)
        self.last_activation = float(
            np.clip(
                (delta - self.gate_threshold) * self.gate_sensitivity,
                0.0,
                1.0,
            )
        )

        rate = self.attack if self.last_activation > self.envelope else self.release
        self.envelope += rate * (self.last_activation - self.envelope)
        return self.continuous.render_frame(state, regions) * self.envelope

    @staticmethod
    def _output_activity_signal(state: FieldState, regions: RegionMasks) -> float:
        if state.phase.shape != regions.shape:
            msg = "state and regions must have matching shapes"
            raise ValueError(msg)
        mask = regions.output
        if not np.any(mask):
            return 0.0
        return float(
            np.mean(state.trace[mask]) + 0.5 * np.mean(1.0 - state.metabolite[mask])
        )


class EventDrivenAudioRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        event_threshold: float = 0.001,
        event_sensitivity: float = 80.0,
        attack: float = 0.6,
        release: float = 0.08,
        hold_frames: int = 10,
        hold_level: float = 0.18,
    ) -> None:
        if event_threshold < 0.0:
            msg = "event_threshold must be non-negative"
            raise ValueError(msg)
        if event_sensitivity <= 0.0:
            msg = "event_sensitivity must be positive"
            raise ValueError(msg)
        if not 0.0 < attack <= 1.0:
            msg = "attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < release <= 1.0:
            msg = "release must be in (0, 1]"
            raise ValueError(msg)
        if hold_frames < 0:
            msg = "hold_frames must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= hold_level <= 1.0:
            msg = "hold_level must be between 0 and 1"
            raise ValueError(msg)

        self.continuous = ContinuousAudioRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.event_threshold = event_threshold
        self.event_sensitivity = event_sensitivity
        self.attack = attack
        self.release = release
        self.hold_frames = hold_frames
        self.hold_level = hold_level
        self.envelope = 0.0
        self.last_activation = 0.0
        self._hold_remaining = 0
        self._previous_features: AudioArray | None = None

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    def render_frame(self, state: FieldState, regions: RegionMasks) -> AudioArray:
        features = self._output_event_features(state, regions)
        if self._previous_features is None:
            self._previous_features = features
            self.last_activation = 0.0
        else:
            event_score = float(np.sum(np.abs(features - self._previous_features)))
            self._previous_features = features
            self.last_activation = float(
                np.clip(
                    (event_score - self.event_threshold) * self.event_sensitivity,
                    0.0,
                    1.0,
                )
            )

        if self.last_activation > 0.0:
            self._hold_remaining = self.hold_frames
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1

        target = self.last_activation
        if self._hold_remaining > 0:
            target = max(target, self.hold_level)

        rate = self.attack if target > self.envelope else self.release
        self.envelope += rate * (target - self.envelope)
        return self.continuous.render_frame(state, regions) * self.envelope

    @staticmethod
    def _output_event_features(state: FieldState, regions: RegionMasks) -> AudioArray:
        if state.phase.shape != regions.shape:
            msg = "state and regions must have matching shapes"
            raise ValueError(msg)
        mask = regions.output
        if not np.any(mask):
            return np.zeros(4, dtype=np.float64)

        phase = state.phase[mask]
        synchrony = float(np.abs(np.mean(np.exp(1j * phase))))
        return np.asarray(
            (
                np.mean(state.trace[mask]),
                np.mean(1.0 - state.metabolite[mask]),
                synchrony,
                np.std(state.frequency[mask]),
            ),
            dtype=np.float64,
        )


class SlopeTriggeredAudioRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        slope_threshold: float = 0.0005,
        slope_sensitivity: float = 220.0,
        attack: float = 0.75,
        release: float = 0.06,
        hold_frames: int = 12,
        hold_level: float = 0.22,
    ) -> None:
        if slope_threshold < 0.0:
            msg = "slope_threshold must be non-negative"
            raise ValueError(msg)
        if slope_sensitivity <= 0.0:
            msg = "slope_sensitivity must be positive"
            raise ValueError(msg)
        if not 0.0 < attack <= 1.0:
            msg = "attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < release <= 1.0:
            msg = "release must be in (0, 1]"
            raise ValueError(msg)
        if hold_frames < 0:
            msg = "hold_frames must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= hold_level <= 1.0:
            msg = "hold_level must be between 0 and 1"
            raise ValueError(msg)

        self.continuous = ContinuousAudioRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.slope_threshold = slope_threshold
        self.slope_sensitivity = slope_sensitivity
        self.attack = attack
        self.release = release
        self.hold_frames = hold_frames
        self.hold_level = hold_level
        self.envelope = 0.0
        self.last_activation = 0.0
        self._hold_remaining = 0
        self._previous_activity: float | None = None

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    def render_frame(self, state: FieldState, regions: RegionMasks) -> AudioArray:
        activity = self._output_activity_signal(state, regions)
        if self._previous_activity is None:
            slope = 0.0
        else:
            slope = max(0.0, activity - self._previous_activity)
        self._previous_activity = activity

        self.last_activation = float(
            np.clip(
                (slope - self.slope_threshold) * self.slope_sensitivity,
                0.0,
                1.0,
            )
        )
        if self.last_activation > 0.0:
            self._hold_remaining = self.hold_frames
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1

        target = self.last_activation
        if self._hold_remaining > 0:
            target = max(target, self.hold_level)

        rate = self.attack if target > self.envelope else self.release
        self.envelope += rate * (target - self.envelope)
        return self.continuous.render_frame(state, regions) * self.envelope

    @staticmethod
    def _output_activity_signal(state: FieldState, regions: RegionMasks) -> float:
        if state.phase.shape != regions.shape:
            msg = "state and regions must have matching shapes"
            raise ValueError(msg)
        mask = regions.output
        if not np.any(mask):
            return 0.0
        phase = state.phase[mask]
        synchrony = float(np.abs(np.mean(np.exp(1j * phase))))
        return float(
            np.mean(state.trace[mask])
            + np.mean(1.0 - state.metabolite[mask])
            + 0.25 * synchrony
        )


class StimulusCoupledAudioRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        input_threshold: float = 0.08,
        input_onset_threshold: float = 0.025,
        retrigger_frames: int = 8,
        response_threshold: float = 0.0004,
        response_sensitivity: float = 260.0,
        response_window_frames: int = 14,
        attack: float = 0.8,
        release: float = 0.07,
        hold_frames: int = 10,
        hold_level: float = 0.2,
    ) -> None:
        if input_threshold < 0.0:
            msg = "input_threshold must be non-negative"
            raise ValueError(msg)
        if input_onset_threshold < 0.0:
            msg = "input_onset_threshold must be non-negative"
            raise ValueError(msg)
        if retrigger_frames < 0:
            msg = "retrigger_frames must be non-negative"
            raise ValueError(msg)
        if response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)
        if response_sensitivity <= 0.0:
            msg = "response_sensitivity must be positive"
            raise ValueError(msg)
        if response_window_frames < 1:
            msg = "response_window_frames must be positive"
            raise ValueError(msg)
        if not 0.0 < attack <= 1.0:
            msg = "attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < release <= 1.0:
            msg = "release must be in (0, 1]"
            raise ValueError(msg)
        if hold_frames < 0:
            msg = "hold_frames must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= hold_level <= 1.0:
            msg = "hold_level must be between 0 and 1"
            raise ValueError(msg)

        self.continuous = ContinuousAudioRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.input_threshold = input_threshold
        self.input_onset_threshold = input_onset_threshold
        self.retrigger_frames = retrigger_frames
        self.response_threshold = response_threshold
        self.response_sensitivity = response_sensitivity
        self.response_window_frames = response_window_frames
        self.attack = attack
        self.release = release
        self.hold_frames = hold_frames
        self.hold_level = hold_level
        self.envelope = 0.0
        self.last_activation = 0.0
        self.stimulus_window = 0.0
        self._window_remaining = 0
        self._retrigger_remaining = 0
        self._hold_remaining = 0
        self._previous_activity: float | None = None
        self._previous_input = 0.0

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    def render_frame(
        self,
        state: FieldState,
        regions: RegionMasks,
        *,
        input_value: float = 0.0,
        response_score: float | None = None,
    ) -> AudioArray:
        input_level = abs(input_value)
        input_rise = max(0.0, input_level - self._previous_input)
        input_started = (
            input_level >= self.input_threshold
            and self._previous_input < self.input_threshold
        )
        input_onset = (
            input_level >= self.input_threshold
            and input_rise >= self.input_onset_threshold
        )
        if self._retrigger_remaining == 0 and (input_started or input_onset):
            self._window_remaining = self.response_window_frames
            self._retrigger_remaining = self.retrigger_frames
        self._previous_input = input_level

        if response_score is None:
            activity = SlopeTriggeredAudioRenderer._output_activity_signal(
                state, regions
            )
            if self._previous_activity is None:
                response_score = 0.0
            else:
                response_score = max(0.0, activity - self._previous_activity)
            self._previous_activity = activity
        else:
            response_score = max(0.0, response_score)

        self.stimulus_window = (
            self._window_remaining / self.response_window_frames
            if self._window_remaining > 0
            else 0.0
        )
        if self._window_remaining > 0:
            self._window_remaining -= 1
            self.last_activation = float(
                np.clip(
                    (response_score - self.response_threshold)
                    * self.response_sensitivity,
                    0.0,
                    1.0,
                )
            )
        else:
            self.last_activation = 0.0
        if self._retrigger_remaining > 0:
            self._retrigger_remaining -= 1

        if self.last_activation > 0.0:
            self._hold_remaining = self.hold_frames
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1

        target = self.last_activation
        if self._hold_remaining > 0:
            target = max(target, self.hold_level)

        rate = self.attack if target > self.envelope else self.release
        self.envelope += rate * (target - self.envelope)
        return self.continuous.render_frame(state, regions) * self.envelope


class VoiceResponseSonificationRenderer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        response_threshold: float = 0.00025,
        response_sensitivity: float = 420.0,
        attack: float = 0.55,
        release: float = 0.05,
        pitch_depth: float = 0.45,
        timbre_depth: float = 0.7,
    ) -> None:
        if response_threshold < 0.0:
            msg = "response_threshold must be non-negative"
            raise ValueError(msg)
        if response_sensitivity <= 0.0:
            msg = "response_sensitivity must be positive"
            raise ValueError(msg)
        if not 0.0 < attack <= 1.0:
            msg = "attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < release <= 1.0:
            msg = "release must be in (0, 1]"
            raise ValueError(msg)
        if pitch_depth < 0.0:
            msg = "pitch_depth must be non-negative"
            raise ValueError(msg)
        if timbre_depth < 0.0:
            msg = "timbre_depth must be non-negative"
            raise ValueError(msg)

        self.continuous = ContinuousAudioRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.response_threshold = response_threshold
        self.response_sensitivity = response_sensitivity
        self.attack = attack
        self.release = release
        self.pitch_depth = pitch_depth
        self.timbre_depth = timbre_depth
        self.envelope = 0.0
        self.last_activation = 0.0
        self._voice_phase = 0.0
        self._brightness = 0.0
        self._previous_activity: float | None = None

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    def render_frame(
        self,
        state: FieldState,
        regions: RegionMasks,
        *,
        response_score: float | None = None,
    ) -> AudioArray:
        if state.phase.shape != regions.shape:
            msg = "state and regions must have matching shapes"
            raise ValueError(msg)
        if response_score is None:
            activity = SlopeTriggeredAudioRenderer._output_activity_signal(
                state, regions
            )
            if self._previous_activity is None:
                response_score = 0.0
            else:
                response_score = max(0.0, activity - self._previous_activity)
            self._previous_activity = activity
        else:
            response_score = max(0.0, response_score)

        self.last_activation = float(
            np.clip(
                (response_score - self.response_threshold) * self.response_sensitivity,
                0.0,
                1.0,
            )
        )
        rate = self.attack if self.last_activation > self.envelope else self.release
        self.envelope += rate * (self.last_activation - self.envelope)

        features = self._output_voice_features(state, regions)
        self._brightness += self.continuous.smoothing * (
            features["brightness"] - self._brightness
        )

        base = self.continuous.render_frame(state, regions)
        response = self._response_voice_frame(features)
        mixed = (0.3 + 0.7 * self.envelope) * base + self.envelope * response
        return np.clip(mixed, -1.0, 1.0).astype(np.float64, copy=False)

    def _response_voice_frame(self, features: dict[str, float]) -> AudioArray:
        samples = np.arange(self.frame_size, dtype=np.float64)
        pitch_shift = 1.0 + self.pitch_depth * (
            0.35 * features["synchrony"]
            + 0.45 * features["trace"]
            + 0.20 * features["frequency_spread"]
        )
        frequency = np.clip(
            self.continuous.carrier_frequency
            * self.continuous.frequency_scale
            * pitch_shift,
            20.0,
            self.continuous.sample_rate / 2.0 - 1.0,
        )
        increment = TAU * frequency / float(self.continuous.sample_rate)
        phase = self._voice_phase + increment * samples
        self._voice_phase = float(
            np.mod(self._voice_phase + increment * self.frame_size, TAU)
        )

        harmonic_mix = np.clip(self._brightness * self.timbre_depth, 0.0, 1.0)
        fundamental = np.sin(phase + features["mean_phase"])
        second = np.sin(2.0 * phase + 0.5 * features["mean_phase"])
        third = np.sin(3.0 * phase - features["mean_phase"])
        voice = (
            (1.0 - 0.45 * harmonic_mix) * fundamental
            + 0.30 * harmonic_mix * second
            + 0.15 * harmonic_mix * third
        )
        return self.continuous.gain * voice

    @staticmethod
    def _output_voice_features(
        state: FieldState,
        regions: RegionMasks,
    ) -> dict[str, float]:
        mask = regions.output
        if not np.any(mask):
            return {
                "synchrony": 0.0,
                "trace": 0.0,
                "metabolite_stress": 0.0,
                "frequency_spread": 0.0,
                "mean_phase": 0.0,
                "brightness": 0.0,
            }

        phase = state.phase[mask]
        order = np.mean(np.exp(1j * phase))
        synchrony = float(np.abs(order))
        trace = float(np.clip(np.mean(state.trace[mask]), 0.0, 1.0))
        metabolite_stress = float(
            np.clip(np.mean(1.0 - state.metabolite[mask]), 0.0, 1.0)
        )
        frequency_spread = float(np.clip(np.std(state.frequency[mask]), 0.0, 1.0))
        brightness = float(
            np.clip(
                0.45 * synchrony + 0.35 * metabolite_stress + 0.20 * frequency_spread,
                0.0,
                1.0,
            )
        )
        return {
            "synchrony": synchrony,
            "trace": trace,
            "metabolite_stress": metabolite_stress,
            "frequency_spread": frequency_spread,
            "mean_phase": float(np.angle(order)),
            "brightness": brightness,
        }


def write_wav(path: str | Path, audio: AudioArray, *, sample_rate: int) -> Path:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.asarray(np.round(clipped * 32767.0), dtype=np.int16)
    with wave.open(str(output_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output_path
