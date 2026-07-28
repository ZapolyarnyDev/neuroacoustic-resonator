from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.audio.synthesis import PatternVoiceSynthesizer
from neuroacoustic_resonator.protocol import SoundProtocolFrame

AudioArray = NDArray[np.float64]

TAU = 2.0 * np.pi


class _ProtocolCarrierRenderer:
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
        self._phase = 0.0
        self._frequency = carrier_frequency
        self._amplitude = 0.0

    def render_frame(self, frame: SoundProtocolFrame) -> AudioArray:
        output = frame.output_region
        target_frequency = float(
            np.clip(
                self.carrier_frequency * self.frequency_scale * output.mean_frequency,
                20.0,
                self.sample_rate / 2.0 - 1.0,
            )
        )
        target_amplitude = (
            self.gain * output.mean_metabolite * (0.2 + 0.8 * output.phase_coherence)
        )
        self._frequency += self.smoothing * (target_frequency - self._frequency)
        self._amplitude += self.smoothing * (target_amplitude - self._amplitude)

        samples = np.arange(self.frame_size, dtype=np.float64)
        increment = TAU * self._frequency / float(self.sample_rate)
        phases = self._phase + increment * samples
        rendered = self._amplitude * np.sin(phases)
        self._phase = float(np.mod(self._phase + increment * self.frame_size, TAU))
        return np.clip(rendered, -1.0, 1.0).astype(np.float64, copy=False)


class ProtocolReferenceRenderer:
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
        min_response_gain: float = 0.0,
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
        if not 0.0 <= min_response_gain <= 1.0:
            msg = "min_response_gain must be between 0 and 1"
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

        self.carrier = _ProtocolCarrierRenderer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
        )
        self.synthesizer = PatternVoiceSynthesizer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            carrier_frequency=carrier_frequency,
            frequency_scale=frequency_scale,
            gain=gain,
            smoothing=smoothing,
            pitch_depth=pitch_depth,
            timbre_depth=timbre_depth,
            pattern_voice_depth=pattern_voice_depth,
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
        self.min_response_gain = min_response_gain
        self.target_response_rms = target_response_rms
        self.energy_normalization_rate = energy_normalization_rate
        self.min_energy_gain = min_energy_gain
        self.max_energy_gain = max_energy_gain
        self.pattern_voice_depth = pattern_voice_depth
        self.envelope = 0.0
        self.last_activation = 0.0
        self.last_pattern_label = "idle"
        self.last_pattern_confidence = 0.0
        self.last_audio_frame = np.zeros(frame_size, dtype=np.float64)
        self._articulation = 0.0
        self._articulation_hold_remaining = 0
        self._previous_response_score = 0.0
        self._previous_activity: float | None = None
        self._energy_gain = 1.0
        self._memory = self._empty_voice_features()

    @property
    def frame_size(self) -> int:
        return self.carrier.frame_size

    @property
    def articulation(self) -> float:
        return self._articulation

    @property
    def energy_gain(self) -> float:
        return self._energy_gain

    def consume(self, frame: SoundProtocolFrame) -> None:
        self.last_audio_frame = self.render_frame(frame)

    def render_frame(self, frame: SoundProtocolFrame) -> AudioArray:
        response_score = self._response_signal(frame)
        self.last_activation = self._soft_activation(response_score)
        rate = self.attack if self.last_activation > self.envelope else self.release
        self.envelope += rate * (self.last_activation - self.envelope)
        self._update_articulation(response_score)

        features = self._protocol_voice_features(frame)
        if frame.active_pattern is None:
            pattern_label = "idle"
            pattern_confidence = 0.0
        else:
            pattern_label = frame.active_pattern.label
            pattern_confidence = frame.active_pattern.confidence
        self.last_pattern_label = pattern_label
        self.last_pattern_confidence = pattern_confidence
        memory_features = self._update_response_memory(features)

        base = self.carrier.render_frame(frame)
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
        response_gain = max(response_gain, self.min_response_gain * self.envelope)
        response_layer = self.response_mix * response_gain * response
        self._update_energy_gain(response_layer, response_gain)
        mixed = background_gain * base + self._energy_gain * response_layer
        return np.clip(mixed, -1.0, 1.0).astype(np.float64, copy=False)

    def _response_signal(self, frame: SoundProtocolFrame) -> float:
        pattern = frame.pattern
        activity = float(
            np.clip(
                pattern.trace_mean
                + pattern.metabolite_stress
                + 0.25 * pattern.phase_order_1,
                0.0,
                1.0,
            )
        )
        change = (
            0.0
            if self._previous_activity is None
            else max(0.0, activity - self._previous_activity)
        )
        self._previous_activity = activity
        active = frame.active_pattern
        sustained = (
            0.0
            if active is None
            else 0.02 * active.intensity * (0.25 + 0.75 * active.confidence)
        )
        novelty = (
            0.0 if active is None or not active.is_novel else 0.01 * active.novelty
        )
        transition = (
            0.0
            if frame.transition is None
            else 0.01 + 0.01 * (0.0 if active is None else active.intensity)
        )
        return max(change, sustained, novelty, transition)

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
        return self.synthesizer.render(
            features,
            pattern_label=pattern_label,
            pattern_confidence=pattern_confidence,
        )

    def _pattern_voice_profile(
        self,
        label: str,
        confidence: float,
    ) -> dict[str, float]:
        return self.synthesizer.profile(label, confidence).__dict__.copy()

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
    def _protocol_voice_features(
        frame: SoundProtocolFrame,
    ) -> dict[str, float]:
        pattern = frame.pattern
        synchrony = pattern.phase_order_1
        trace = float(np.clip(pattern.trace_mean, 0.0, 1.0))
        frequency_spread = float(np.clip(pattern.frequency_spread, 0.0, 1.0))
        frequency_mean = float(np.clip(pattern.frequency_mean, 0.0, 2.0) / 2.0)
        phase_spread = float(np.clip(1.0 - synchrony, 0.0, 1.0))
        brightness = float(
            np.clip(
                0.28 * synchrony
                + 0.24 * pattern.metabolite_stress
                + 0.20 * frequency_spread
                + 0.18 * pattern.trace_contrast
                + 0.10 * frequency_mean
                + 0.08 * pattern.phase_order_2,
                0.0,
                1.0,
            )
        )
        roughness = float(
            np.clip(
                0.45 * phase_spread
                + 0.25 * pattern.metabolite_contrast
                + 0.20 * pattern.trace_contrast
                + 0.10 * frequency_spread
                + 0.10 * pattern.phase_order_3,
                0.0,
                1.0,
            )
        )
        return {
            "synchrony": synchrony,
            "trace": trace,
            "metabolite_stress": pattern.metabolite_stress,
            "metabolite_contrast": pattern.metabolite_contrast,
            "frequency_spread": frequency_spread,
            "frequency_mean": frequency_mean,
            "mean_phase": 0.0,
            "phase_angle_2": 0.0,
            "phase_angle_3": 0.0,
            "phase_order_2": pattern.phase_order_2,
            "phase_order_3": pattern.phase_order_3,
            "phase_spread": phase_spread,
            "trace_phase_lock": pattern.trace_phase_lock,
            "metabolite_phase_lock": pattern.metabolite_phase_lock,
            "trace_contrast": pattern.trace_contrast,
            "brightness": brightness,
            "roughness": roughness,
        }
