import wave

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.audio_output import render_output_frame, write_wav


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
