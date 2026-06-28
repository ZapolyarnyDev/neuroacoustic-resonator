from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.analysis.output_patterns import classify_output_pattern
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
        response_sensitivity: float = 220.0,
        attack: float = 0.55,
        release: float = 0.05,
        pitch_depth: float = 0.45,
        timbre_depth: float = 0.7,
        background_level: float = 0.03,
        background_response_level: float = 0.16,
        response_mix: float = 1.35,
        response_memory: float = 0.35,
        response_memory_decay: float = 0.03,
        articulation_attack: float = 0.9,
        articulation_release: float = 0.2,
        articulation_hold_frames: int = 5,
        articulation_floor: float = 0.02,
        target_response_rms: float = 0.1,
        energy_normalization_rate: float = 0.06,
        min_energy_gain: float = 0.65,
        max_energy_gain: float = 1.8,
        pattern_voice_depth: float = 0.55,
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
        if background_level < 0.0:
            msg = "background_level must be non-negative"
            raise ValueError(msg)
        if background_response_level < background_level:
            msg = "background_response_level must be at least background_level"
            raise ValueError(msg)
        if response_mix < 0.0:
            msg = "response_mix must be non-negative"
            raise ValueError(msg)
        if response_memory < 0.0:
            msg = "response_memory must be non-negative"
            raise ValueError(msg)
        if not 0.0 < response_memory_decay <= 1.0:
            msg = "response_memory_decay must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < articulation_attack <= 1.0:
            msg = "articulation_attack must be in (0, 1]"
            raise ValueError(msg)
        if not 0.0 < articulation_release <= 1.0:
            msg = "articulation_release must be in (0, 1]"
            raise ValueError(msg)
        if articulation_hold_frames < 0:
            msg = "articulation_hold_frames must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= articulation_floor <= 1.0:
            msg = "articulation_floor must be between 0 and 1"
            raise ValueError(msg)
        if target_response_rms <= 0.0:
            msg = "target_response_rms must be positive"
            raise ValueError(msg)
        if not 0.0 < energy_normalization_rate <= 1.0:
            msg = "energy_normalization_rate must be in (0, 1]"
            raise ValueError(msg)
        if min_energy_gain <= 0.0:
            msg = "min_energy_gain must be positive"
            raise ValueError(msg)
        if max_energy_gain < min_energy_gain:
            msg = "max_energy_gain must be at least min_energy_gain"
            raise ValueError(msg)
        if pattern_voice_depth < 0.0:
            msg = "pattern_voice_depth must be non-negative"
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
        self.background_level = background_level
        self.background_response_level = background_response_level
        self.response_mix = response_mix
        self.response_memory = response_memory
        self.response_memory_decay = response_memory_decay
        self.articulation_attack = articulation_attack
        self.articulation_release = articulation_release
        self.articulation_hold_frames = articulation_hold_frames
        self.articulation_floor = articulation_floor
        self.target_response_rms = target_response_rms
        self.energy_normalization_rate = energy_normalization_rate
        self.min_energy_gain = min_energy_gain
        self.max_energy_gain = max_energy_gain
        self.pattern_voice_depth = pattern_voice_depth
        self.envelope = 0.0
        self.last_activation = 0.0
        self.last_pattern_label = "idle"
        self.last_pattern_confidence = 0.0
        self._articulation = 0.0
        self._articulation_hold_remaining = 0
        self._previous_response_score = 0.0
        self._energy_gain = 1.0
        self._voice_phase = 0.0
        self._brightness = 0.0
        self._roughness = 0.0
        self._vibrato_phase = 0.0
        self._memory = self._empty_voice_features()
        self._previous_activity: float | None = None

    @property
    def frame_size(self) -> int:
        return self.continuous.frame_size

    @property
    def articulation(self) -> float:
        return self._articulation

    @property
    def energy_gain(self) -> float:
        return self._energy_gain

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

        self.last_activation = self._soft_activation(response_score)
        rate = self.attack if self.last_activation > self.envelope else self.release
        self.envelope += rate * (self.last_activation - self.envelope)
        self._update_articulation(response_score)

        features = self._output_voice_features(state, regions)
        pattern_label, pattern_confidence = classify_output_pattern(features)
        self.last_pattern_label = pattern_label
        self.last_pattern_confidence = pattern_confidence
        self._brightness += self.continuous.smoothing * (
            features["brightness"] - self._brightness
        )
        self._roughness += self.continuous.smoothing * (
            features["roughness"] - self._roughness
        )
        memory_features = self._update_response_memory(features)

        base = self.continuous.render_frame(state, regions)
        response = self._response_voice_frame(
            memory_features,
            pattern_label=pattern_label,
            pattern_confidence=pattern_confidence,
        )
        background_gain = (
            self.background_level
            + (self.background_response_level - self.background_level) * self.envelope
        )
        response_gain = self.envelope * (
            self.articulation_floor
            + (1.0 - self.articulation_floor) * self._articulation
        )
        response_layer = self.response_mix * response_gain * response
        self._update_energy_gain(response_layer, response_gain)
        mixed = background_gain * base + self._energy_gain * response_layer
        return np.clip(mixed, -1.0, 1.0).astype(np.float64, copy=False)

    def _update_articulation(self, response_score: float) -> None:
        response_rise = max(0.0, response_score - self._previous_response_score)
        self._previous_response_score = response_score
        event_activation = max(
            0.25 * self.last_activation,
            self._soft_activation(response_rise * 2.5),
        )
        if event_activation > 0.0:
            self._articulation_hold_remaining = self.articulation_hold_frames
        elif self._articulation_hold_remaining > 0:
            self._articulation_hold_remaining -= 1

        target = event_activation
        if self._articulation_hold_remaining > 0:
            target = max(target, self.articulation_floor + 0.18 * self.envelope)

        rate = (
            self.articulation_attack
            if target > self._articulation
            else self.articulation_release
        )
        self._articulation += rate * (target - self._articulation)

    def _update_energy_gain(
        self, response_layer: AudioArray, response_gain: float
    ) -> None:
        if response_gain <= 1e-6:
            target_gain = 1.0
        else:
            rms = float(np.sqrt(np.mean(response_layer * response_layer)))
            if rms <= 1e-9:
                target_gain = self.max_energy_gain
            else:
                target_gain = self.target_response_rms / rms
            target_gain = float(
                np.clip(target_gain, self.min_energy_gain, self.max_energy_gain)
            )
        self._energy_gain += self.energy_normalization_rate * (
            target_gain - self._energy_gain
        )

    def _response_voice_frame(
        self,
        features: dict[str, float],
        *,
        pattern_label: str = "mixed",
        pattern_confidence: float = 0.0,
    ) -> AudioArray:
        samples = np.arange(self.frame_size, dtype=np.float64)
        pattern = self._pattern_voice_profile(pattern_label, pattern_confidence)
        phase_spread = features["phase_spread"]
        trace_contrast = features["trace_contrast"]
        metabolite_contrast = features["metabolite_contrast"]
        phase_order_2 = features["phase_order_2"]
        phase_order_3 = features["phase_order_3"]
        trace_phase_lock = features["trace_phase_lock"]
        metabolite_phase_lock = features["metabolite_phase_lock"]
        base_pitch_shift = pattern["pitch"] * (
            1.0
            + self.pitch_depth
            * (
                0.25 * features["synchrony"]
                + 0.30 * features["trace"]
                + 0.20 * features["frequency_mean"]
                + 0.15 * features["frequency_spread"]
                - 0.10 * features["metabolite_stress"]
                + 0.08 * phase_order_2
                - 0.06 * phase_order_3
            )
        )
        vibrato_rate = (
            3.0 + 5.0 * phase_spread + 2.0 * trace_contrast + 2.5 * trace_phase_lock
        )
        vibrato_increment = TAU * vibrato_rate / float(self.continuous.sample_rate)
        vibrato_phase = self._vibrato_phase + vibrato_increment * samples
        self._vibrato_phase = float(
            np.mod(self._vibrato_phase + vibrato_increment * self.frame_size, TAU)
        )
        vibrato_depth = 0.004 + 0.035 * phase_spread + 0.02 * metabolite_contrast
        pitch_shift = base_pitch_shift * (1.0 + vibrato_depth * np.sin(vibrato_phase))
        frequency = np.clip(
            self.continuous.carrier_frequency
            * self.continuous.frequency_scale
            * pitch_shift,
            20.0,
            self.continuous.sample_rate / 2.0 - 1.0,
        )
        increment = TAU * frequency / float(self.continuous.sample_rate)
        phase = self._voice_phase + np.cumsum(increment)
        self._voice_phase = float(
            np.mod(self._voice_phase + float(np.sum(increment)), TAU)
        )

        harmonic_mix = np.clip(
            (self._brightness + 0.25 * phase_order_2 + pattern["harmonic_bias"])
            * self.timbre_depth,
            0.0,
            1.0,
        )
        rough_mix = np.clip(
            (
                self._roughness
                + 0.20 * phase_order_3
                + 0.15 * metabolite_phase_lock
                + pattern["rough_bias"]
            )
            * self.timbre_depth,
            0.0,
            1.0,
        )
        formant_phase = features["mean_phase"]
        second_phase = features["phase_angle_2"]
        third_phase = features["phase_angle_3"]
        fundamental = np.sin(phase + formant_phase)
        second = np.sin(2.0 * phase + 0.5 * formant_phase + second_phase)
        third = np.sin(3.0 * phase - formant_phase + third_phase)
        sub = np.sin(0.5 * phase + formant_phase - third_phase)
        fourth = np.sin(4.0 * phase + 1.7 * formant_phase - second_phase)
        ratio_a = 2.31 + 0.55 * phase_order_2 + 0.21 * trace_phase_lock
        ratio_b = 4.63 + 0.71 * phase_order_3 + 0.33 * metabolite_phase_lock
        inharmonic = np.sin(
            ratio_a * phase + 0.7 * formant_phase + second_phase
        ) * np.sin(ratio_b * phase - 0.3 * formant_phase + third_phase)
        sparse_focus = np.clip(0.35 + 0.65 * phase_order_2, 0.0, 1.0)
        rough_focus = np.clip(1.0 - 0.45 * sparse_focus, 0.0, 1.0)
        voice = (
            (1.0 - 0.32 * harmonic_mix) * fundamental
            + (0.10 + 0.10 * trace_contrast + 0.08 * phase_order_2)
            * harmonic_mix
            * sparse_focus
            * second
            + (0.04 + 0.10 * metabolite_contrast + 0.08 * phase_order_3)
            * harmonic_mix
            * (0.65 + 0.35 * phase_spread)
            * third
            + 0.035 * rough_mix * rough_focus * fourth
            + (0.045 + 0.065 * phase_spread) * rough_mix * rough_focus * inharmonic
            + pattern["sub_mix"] * sub
        )
        pulse = 1.0 + pattern["pulse_depth"] * np.sin(
            (1.0 + pattern["pulse_rate"]) * phase + second_phase
        )
        normalization = (
            1.0 + 0.32 * harmonic_mix + 0.16 * rough_mix + pattern["sub_mix"]
        )
        voice = voice * pulse
        return self.continuous.gain * voice / normalization

    def _pattern_voice_profile(
        self,
        label: str,
        confidence: float,
    ) -> dict[str, float]:
        depth = self.pattern_voice_depth * float(np.clip(confidence, 0.0, 1.0))
        base = {
            "pitch": 1.0,
            "harmonic_bias": 0.0,
            "rough_bias": 0.0,
            "sub_mix": 0.0,
            "pulse_depth": 0.0,
            "pulse_rate": 0.0,
        }
        profiles = {
            "coherent": {
                "pitch": 1.0 + 0.10 * depth,
                "harmonic_bias": 0.12 * depth,
                "rough_bias": -0.10 * depth,
            },
            "split": {
                "pitch": 0.94 - 0.08 * depth,
                "harmonic_bias": 0.22 * depth,
                "sub_mix": 0.16 * depth,
                "pulse_depth": 0.08 * depth,
                "pulse_rate": 0.5,
            },
            "triadic": {
                "pitch": 1.08 + 0.12 * depth,
                "harmonic_bias": 0.12 * depth,
                "rough_bias": 0.08 * depth,
                "pulse_depth": 0.05 * depth,
                "pulse_rate": 2.0,
            },
            "diffuse": {
                "pitch": 0.98,
                "rough_bias": 0.30 * depth,
                "sub_mix": 0.04 * depth,
                "pulse_depth": 0.12 * depth,
                "pulse_rate": 3.0,
            },
            "imprinted": {
                "pitch": 0.90 - 0.04 * depth,
                "harmonic_bias": 0.06 * depth,
                "rough_bias": 0.12 * depth,
                "sub_mix": 0.11 * depth,
                "pulse_depth": 0.04 * depth,
                "pulse_rate": 1.0,
            },
        }
        for key, value in profiles.get(label, {}).items():
            base[key] = value
        return base

    def _soft_activation(self, response_score: float) -> float:
        scaled = max(0.0, response_score - self.response_threshold)
        return float(1.0 - np.exp(-scaled * self.response_sensitivity))

    def _update_response_memory(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        imprint = self.response_memory * self.envelope
        decay = self.response_memory_decay
        mixed: dict[str, float] = {}
        for key, value in features.items():
            remembered = self._memory[key] * (1.0 - decay)
            remembered += imprint * (value - remembered)
            self._memory[key] = remembered
            mixed[key] = float(
                np.clip(
                    value + self.response_memory * self._memory[key],
                    -TAU,
                    TAU,
                )
            )
        return mixed

    @staticmethod
    def _empty_voice_features() -> dict[str, float]:
        return {
            "synchrony": 0.0,
            "trace": 0.0,
            "metabolite_stress": 0.0,
            "metabolite_contrast": 0.0,
            "frequency_spread": 0.0,
            "frequency_mean": 0.0,
            "mean_phase": 0.0,
            "phase_angle_2": 0.0,
            "phase_angle_3": 0.0,
            "phase_order_2": 0.0,
            "phase_order_3": 0.0,
            "phase_spread": 0.0,
            "trace_phase_lock": 0.0,
            "metabolite_phase_lock": 0.0,
            "trace_contrast": 0.0,
            "brightness": 0.0,
            "roughness": 0.0,
        }

    @staticmethod
    def _output_voice_features(
        state: FieldState,
        regions: RegionMasks,
    ) -> dict[str, float]:
        mask = regions.output
        if not np.any(mask):
            return VoiceResponseSonificationRenderer._empty_voice_features()
        phase = state.phase[mask]
        order = np.mean(np.exp(1j * phase))
        second_order = np.mean(np.exp(2j * phase))
        third_order = np.mean(np.exp(3j * phase))
        synchrony = float(np.abs(order))
        trace = float(np.clip(np.mean(state.trace[mask]), 0.0, 1.0))
        trace_contrast = float(np.clip(np.std(state.trace[mask]), 0.0, 1.0))
        metabolite_stress = float(
            np.clip(np.mean(1.0 - state.metabolite[mask]), 0.0, 1.0)
        )
        metabolite_contrast = float(
            np.clip(np.std(1.0 - state.metabolite[mask]), 0.0, 1.0)
        )
        frequency_spread = float(np.clip(np.std(state.frequency[mask]), 0.0, 1.0))
        frequency_mean = float(np.clip(np.mean(state.frequency[mask]), 0.0, 2.0) / 2.0)
        phase_spread = float(np.clip(1.0 - synchrony, 0.0, 1.0))
        phase_order_2 = float(np.abs(second_order))
        phase_order_3 = float(np.abs(third_order))
        trace_weights = np.clip(state.trace[mask], 0.0, None)
        trace_weight_sum = float(np.sum(trace_weights))
        if trace_weight_sum > 1e-12:
            trace_phase_lock = float(
                np.abs(np.sum(trace_weights * np.exp(1j * phase)) / trace_weight_sum)
            )
        else:
            trace_phase_lock = 0.0
        metabolite_weights = np.clip(1.0 - state.metabolite[mask], 0.0, None)
        metabolite_weight_sum = float(np.sum(metabolite_weights))
        if metabolite_weight_sum > 1e-12:
            metabolite_phase_lock = float(
                np.abs(
                    np.sum(metabolite_weights * np.exp(1j * phase))
                    / metabolite_weight_sum
                )
            )
        else:
            metabolite_phase_lock = 0.0
        brightness = float(
            np.clip(
                0.28 * synchrony
                + 0.24 * metabolite_stress
                + 0.20 * frequency_spread
                + 0.18 * trace_contrast
                + 0.10 * frequency_mean
                + 0.08 * phase_order_2,
                0.0,
                1.0,
            )
        )
        roughness = float(
            np.clip(
                0.45 * phase_spread
                + 0.25 * metabolite_contrast
                + 0.20 * trace_contrast
                + 0.10 * frequency_spread
                + 0.10 * phase_order_3,
                0.0,
                1.0,
            )
        )
        return {
            "synchrony": synchrony,
            "trace": trace,
            "metabolite_stress": metabolite_stress,
            "metabolite_contrast": metabolite_contrast,
            "frequency_spread": frequency_spread,
            "frequency_mean": frequency_mean,
            "mean_phase": float(np.angle(order)),
            "phase_angle_2": float(np.angle(second_order)),
            "phase_angle_3": float(np.angle(third_order)),
            "phase_order_2": phase_order_2,
            "phase_order_3": phase_order_3,
            "phase_spread": phase_spread,
            "trace_phase_lock": trace_phase_lock,
            "metabolite_phase_lock": metabolite_phase_lock,
            "trace_contrast": trace_contrast,
            "brightness": brightness,
            "roughness": roughness,
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
