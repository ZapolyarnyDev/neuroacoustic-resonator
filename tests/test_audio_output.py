import wave

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.audio_output import (
    ContinuousAudioRenderer,
    GatedAudioRenderer,
    render_output_frame,
    write_wav,
)


def test_render_output_frame_returns_bounded_audio() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)

    audio = render_output_frame(field.state, regions, frame_size=128, gain=0.5)

    assert audio.shape == (128,)
    assert audio.dtype == np.float64
    assert np.all(np.isfinite(audio))
    assert np.all((-1.0 <= audio) & (audio <= 1.0))


def test_render_output_frame_depends_on_output_region_phase() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    state = field.state
    shifted_state = type(state)(
        phase=state.phase.copy(),
        frequency=state.frequency.copy(),
        metabolite=state.metabolite.copy(),
        coupling=state.coupling.copy(),
        trace=state.trace.copy(),
    )
    shifted_state.phase[regions.output] += np.pi / 2.0

    audio = render_output_frame(state, regions, frame_size=64)
    shifted_audio = render_output_frame(shifted_state, regions, frame_size=64)

    assert not np.allclose(audio, shifted_audio)


def test_render_output_frame_rejects_shape_mismatch() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(5)

    with pytest.raises(ValueError, match="matching shapes"):
        render_output_frame(field.state, regions)


def test_write_wav_creates_pcm_file(tmp_path) -> None:
    audio = np.linspace(-1.0, 1.0, 16)

    output_path = write_wav(tmp_path / "demo.wav", audio, sample_rate=8_000)

    assert output_path.exists()
    with wave.open(str(output_path), "rb") as stream:
        assert stream.getframerate() == 8_000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() == 16


def test_continuous_audio_renderer_returns_bounded_frames() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = ContinuousAudioRenderer(sample_rate=8_000, frame_size=64, gain=0.5)

    first = renderer.render_frame(field.state, regions)
    field.step()
    second = renderer.render_frame(field.state, regions)

    assert first.shape == (64,)
    assert second.shape == (64,)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    assert np.all((-1.0 <= first) & (first <= 1.0))
    assert np.all((-1.0 <= second) & (second <= 1.0))


def test_continuous_audio_renderer_maintains_phase_between_frames() -> None:
    field = OscillatorField(
        FieldConfig(
            size=4,
            seed=1,
            base_frequency=1.0,
            frequency_spread=0.0,
        )
    )
    regions = RegionMasks.from_size(4)
    renderer = ContinuousAudioRenderer(
        sample_rate=8_000,
        frame_size=80,
        carrier_frequency=100.0,
        gain=0.5,
        smoothing=1.0,
    )

    first = renderer.render_frame(field.state, regions)
    second = renderer.render_frame(field.state, regions)

    assert abs(second[0] - first[-1]) < 0.1


def test_continuous_audio_renderer_rejects_invalid_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        ContinuousAudioRenderer(smoothing=0.0)


def test_gated_audio_renderer_suppresses_steady_state() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = GatedAudioRenderer(sample_rate=8_000, frame_size=64, gain=0.5)

    first = renderer.render_frame(field.state, regions)
    second = renderer.render_frame(field.state, regions)

    assert first.shape == (64,)
    assert second.shape == (64,)
    assert np.max(np.abs(first)) == 0.0
    assert np.max(np.abs(second)) == 0.0


def test_gated_audio_renderer_opens_on_output_activity_change() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = GatedAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        gate_threshold=0.0,
        gate_sensitivity=100.0,
        attack=1.0,
    )
    renderer.render_frame(field.state, regions)
    state = field.state
    state.trace[regions.output] += 0.1

    audio = renderer.render_frame(state, regions)

    assert renderer.envelope > 0.0
    assert np.max(np.abs(audio)) > 0.0


def test_gated_audio_renderer_rejects_invalid_gate_threshold() -> None:
    with pytest.raises(ValueError, match="gate_threshold"):
        GatedAudioRenderer(gate_threshold=-0.1)
