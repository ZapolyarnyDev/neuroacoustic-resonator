from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AudioArray = NDArray[np.float64]
TAU = 2.0 * np.pi


@dataclass(frozen=True)
class PatternVoiceProfile:
    pitch: float = 1.0
    harmonic_bias: float = 0.0
    rough_bias: float = 0.0
    sub_mix: float = 0.0
    pulse_depth: float = 0.0
    pulse_rate: float = 0.0
    noise_mix: float = 0.0
    fold_mix: float = 0.0


class PatternVoiceSynthesizer:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        frame_size: int = 512,
        carrier_frequency: float = 220.0,
        frequency_scale: float = 1.0,
        gain: float = 0.2,
        smoothing: float = 0.2,
        pitch_depth: float = 0.45,
        timbre_depth: float = 0.7,
        pattern_voice_depth: float = 0.55,
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
        if pitch_depth < 0.0:
            msg = "pitch_depth must be non-negative"
            raise ValueError(msg)
        if timbre_depth < 0.0:
            msg = "timbre_depth must be non-negative"
            raise ValueError(msg)
        if pattern_voice_depth < 0.0:
            msg = "pattern_voice_depth must be non-negative"
            raise ValueError(msg)

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.carrier_frequency = carrier_frequency
        self.frequency_scale = frequency_scale
        self.gain = gain
        self.smoothing = smoothing
        self.pitch_depth = pitch_depth
        self.timbre_depth = timbre_depth
        self.pattern_voice_depth = pattern_voice_depth
        self._voice_phase = 0.0
        self._vibrato_phase = 0.0
        self._noise_phase = 0.0
        self._brightness = 0.0
        self._roughness = 0.0

    def render(
        self,
        features: dict[str, float],
        *,
        pattern_label: str = "mixed",
        pattern_confidence: float = 0.0,
    ) -> AudioArray:
        self._brightness += self.smoothing * (features["brightness"] - self._brightness)
        self._roughness += self.smoothing * (features["roughness"] - self._roughness)

        samples = np.arange(self.frame_size, dtype=np.float64)
        profile = self.profile(pattern_label, pattern_confidence)
        phase_spread = features["phase_spread"]
        trace_contrast = features["trace_contrast"]
        metabolite_contrast = features["metabolite_contrast"]
        phase_order_2 = features["phase_order_2"]
        phase_order_3 = features["phase_order_3"]
        trace_phase_lock = features["trace_phase_lock"]
        metabolite_phase_lock = features["metabolite_phase_lock"]

        base_pitch_shift = profile.pitch * (
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
        vibrato_increment = TAU * vibrato_rate / float(self.sample_rate)
        vibrato_phase = self._vibrato_phase + vibrato_increment * samples
        self._vibrato_phase = float(
            np.mod(self._vibrato_phase + vibrato_increment * self.frame_size, TAU)
        )
        vibrato_depth = 0.004 + 0.035 * phase_spread + 0.02 * metabolite_contrast
        pitch_shift = base_pitch_shift * (1.0 + vibrato_depth * np.sin(vibrato_phase))
        frequency = np.clip(
            self.carrier_frequency * self.frequency_scale * pitch_shift,
            20.0,
            self.sample_rate / 2.0 - 1.0,
        )
        increment = TAU * frequency / float(self.sample_rate)
        phase = self._voice_phase + np.cumsum(increment)
        self._voice_phase = float(
            np.mod(self._voice_phase + float(np.sum(increment)), TAU)
        )

        harmonic_mix = np.clip(
            (self._brightness + 0.25 * phase_order_2 + profile.harmonic_bias)
            * self.timbre_depth,
            0.0,
            1.0,
        )
        rough_mix = np.clip(
            (
                self._roughness
                + 0.20 * phase_order_3
                + 0.15 * metabolite_phase_lock
                + profile.rough_bias
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
        noise = self._deterministic_noise(samples, phase, formant_phase)

        sparse_focus = np.clip(0.35 + 0.65 * phase_order_2, 0.0, 1.0)
        rough_focus = np.clip(1.0 - 0.45 * sparse_focus, 0.0, 1.0)
        harmonic_voice = (
            (1.0 - 0.32 * harmonic_mix) * fundamental
            + (0.10 + 0.10 * trace_contrast + 0.08 * phase_order_2)
            * harmonic_mix
            * sparse_focus
            * second
            + (0.04 + 0.10 * metabolite_contrast + 0.08 * phase_order_3)
            * harmonic_mix
            * (0.65 + 0.35 * phase_spread)
            * third
        )
        rough_voice = (
            0.035 * rough_mix * rough_focus * fourth
            + (0.045 + 0.065 * phase_spread) * rough_mix * rough_focus * inharmonic
            + profile.noise_mix * noise
        )
        voice = harmonic_voice + rough_voice + profile.sub_mix * sub
        if profile.fold_mix > 0.0:
            folded = np.tanh((1.6 + 3.2 * profile.fold_mix) * voice)
            voice = (1.0 - profile.fold_mix) * voice + profile.fold_mix * folded

        pulse = 1.0 + profile.pulse_depth * np.sin(
            (1.0 + profile.pulse_rate) * phase + second_phase
        )
        normalization = (
            1.0
            + 0.32 * harmonic_mix
            + 0.16 * rough_mix
            + profile.sub_mix
            + profile.noise_mix
        )
        return self.gain * voice * pulse / normalization

    def profile(self, label: str, confidence: float) -> PatternVoiceProfile:
        depth = self.pattern_voice_depth * float(np.clip(confidence, 0.0, 1.0))
        profiles = {
            "coherent": PatternVoiceProfile(
                pitch=1.0 + 0.18 * depth,
                harmonic_bias=0.22 * depth,
                rough_bias=-0.18 * depth,
            ),
            "split": PatternVoiceProfile(
                pitch=0.86 - 0.12 * depth,
                harmonic_bias=0.30 * depth,
                sub_mix=0.32 * depth,
                pulse_depth=0.18 * depth,
                pulse_rate=0.5,
                fold_mix=0.10 * depth,
            ),
            "triadic": PatternVoiceProfile(
                pitch=1.15 + 0.18 * depth,
                harmonic_bias=0.20 * depth,
                rough_bias=0.10 * depth,
                pulse_depth=0.12 * depth,
                pulse_rate=2.0,
                fold_mix=0.06 * depth,
            ),
            "diffuse": PatternVoiceProfile(
                pitch=0.96,
                rough_bias=0.46 * depth,
                sub_mix=0.06 * depth,
                pulse_depth=0.20 * depth,
                pulse_rate=3.5,
                noise_mix=0.24 * depth,
                fold_mix=0.08 * depth,
            ),
            "imprinted": PatternVoiceProfile(
                pitch=0.82 - 0.08 * depth,
                harmonic_bias=0.10 * depth,
                rough_bias=0.16 * depth,
                sub_mix=0.24 * depth,
                pulse_depth=0.08 * depth,
                pulse_rate=1.0,
            ),
        }
        return profiles.get(label, PatternVoiceProfile())

    def _deterministic_noise(
        self,
        samples: AudioArray,
        phase: AudioArray,
        formant_phase: float,
    ) -> AudioArray:
        noise_increment = TAU * 37.0 / float(self.sample_rate)
        noise_phase = self._noise_phase + noise_increment * samples
        self._noise_phase = float(
            np.mod(self._noise_phase + noise_increment * self.frame_size, TAU)
        )
        raw = (
            np.sin(12.9898 * phase + 78.233 * np.sin(noise_phase + formant_phase))
            * 43758.5453
        )
        return 2.0 * (raw - np.floor(raw)) - 1.0
