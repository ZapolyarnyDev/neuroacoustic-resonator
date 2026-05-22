import wave

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.audio.output import (
    ContinuousAudioRenderer,
    EventDrivenAudioRenderer,
    GatedAudioRenderer,
    SlopeTriggeredAudioRenderer,
    StimulusCoupledAudioRenderer,
    VoiceResponseSonificationRenderer,
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


def test_event_driven_audio_renderer_starts_silent() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = EventDrivenAudioRenderer(sample_rate=8_000, frame_size=64, gain=0.5)

    audio = renderer.render_frame(field.state, regions)

    assert audio.shape == (64,)
    assert renderer.envelope == 0.0
    assert np.max(np.abs(audio)) == 0.0


def test_event_driven_audio_renderer_opens_on_feature_change() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = EventDrivenAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        event_threshold=0.0,
        event_sensitivity=100.0,
        attack=1.0,
    )
    renderer.render_frame(field.state, regions)
    state = field.state
    state.trace[regions.output] += 0.1

    audio = renderer.render_frame(state, regions)

    assert renderer.last_activation > 0.0
    assert renderer.envelope > 0.0
    assert np.max(np.abs(audio)) > 0.0


def test_event_driven_audio_renderer_holds_after_event() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = EventDrivenAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        event_threshold=0.0,
        event_sensitivity=100.0,
        attack=1.0,
        hold_frames=2,
        hold_level=0.25,
    )
    renderer.render_frame(field.state, regions)
    state = field.state
    state.trace[regions.output] += 0.1
    renderer.render_frame(state, regions)

    held = renderer.render_frame(state, regions)

    assert renderer.last_activation == 0.0
    assert renderer.envelope >= 0.25
    assert np.max(np.abs(held)) > 0.0


def test_event_driven_audio_renderer_rejects_invalid_hold_frames() -> None:
    with pytest.raises(ValueError, match="hold_frames"):
        EventDrivenAudioRenderer(hold_frames=-1)


def test_slope_triggered_audio_renderer_opens_on_rising_output_activity() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = SlopeTriggeredAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        slope_threshold=0.0,
        slope_sensitivity=100.0,
        attack=1.0,
    )
    renderer.render_frame(field.state, regions)
    state = field.state
    state.trace[regions.output] += 0.1

    audio = renderer.render_frame(state, regions)

    assert renderer.last_activation > 0.0
    assert renderer.envelope > 0.0
    assert np.max(np.abs(audio)) > 0.0


def test_slope_triggered_audio_renderer_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="slope_threshold"):
        SlopeTriggeredAudioRenderer(slope_threshold=-0.1)


def test_stimulus_coupled_audio_renderer_requires_input_before_response() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = StimulusCoupledAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        input_threshold=0.1,
        input_onset_threshold=0.0,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
    )
    renderer.render_frame(field.state, regions)
    state = field.state
    state.trace[regions.output] += 0.1

    silent = renderer.render_frame(state, regions, input_value=0.0)
    state.trace[regions.output] += 0.1
    opened = renderer.render_frame(state, regions, input_value=0.2)

    assert renderer.stimulus_window > 0.0
    assert np.max(np.abs(silent)) == 0.0
    assert np.max(np.abs(opened)) > 0.0


def test_stimulus_coupled_audio_renderer_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="response_window_frames"):
        StimulusCoupledAudioRenderer(response_window_frames=0)


def test_stimulus_coupled_audio_renderer_uses_onset_not_sustained_input() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = StimulusCoupledAudioRenderer(
        sample_rate=8_000,
        frame_size=64,
        input_threshold=0.1,
        input_onset_threshold=0.05,
        retrigger_frames=0,
        response_window_frames=3,
    )

    renderer.render_frame(field.state, regions, input_value=0.2)
    first_window = renderer.stimulus_window
    renderer.render_frame(field.state, regions, input_value=0.2)
    second_window = renderer.stimulus_window
    renderer.render_frame(field.state, regions, input_value=0.2)
    third_window = renderer.stimulus_window

    assert first_window == pytest.approx(1.0)
    assert second_window < first_window
    assert third_window < second_window


def test_voice_response_sonification_uses_response_score() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
    )

    quiet = renderer.render_frame(field.state, regions, response_score=0.0)
    active = renderer.render_frame(field.state, regions, response_score=0.02)

    assert renderer.last_activation > 0.0
    assert renderer.envelope > 0.0
    assert np.max(np.abs(active)) > np.max(np.abs(quiet))
    assert np.all((-1.0 <= active) & (active <= 1.0))


def test_voice_response_sonification_softens_activation_before_ceiling() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=64,
        response_threshold=0.0,
        response_sensitivity=220.0,
        attack=1.0,
    )

    renderer.render_frame(field.state, regions, response_score=0.001)
    moderate_activation = renderer.last_activation
    renderer.render_frame(field.state, regions, response_score=0.01)

    assert 0.0 < moderate_activation < 0.5
    assert moderate_activation < renderer.last_activation < 1.0


def test_voice_response_sonification_keeps_idle_background_quiet() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        background_level=0.02,
        background_response_level=0.25,
    )

    quiet = renderer.render_frame(field.state, regions, response_score=0.0)
    active = renderer.render_frame(field.state, regions, response_score=0.02)

    assert np.max(np.abs(quiet)) < 0.02
    assert np.max(np.abs(active)) > np.max(np.abs(quiet)) * 4.0


def test_voice_response_sonification_changes_with_output_state() -> None:
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=64,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
    )
    renderer.render_frame(field.state, regions, response_score=0.02)
    before = renderer.render_frame(field.state, regions, response_score=0.02)
    changed_state = type(field.state)(
        phase=field.state.phase.copy(),
        frequency=field.state.frequency.copy(),
        metabolite=field.state.metabolite.copy(),
        coupling=field.state.coupling.copy(),
        trace=field.state.trace.copy(),
    )
    changed_state.trace[regions.output] += 0.5
    changed_state.phase[regions.output] += np.pi / 3.0

    after = renderer.render_frame(changed_state, regions, response_score=0.02)

    assert not np.allclose(before, after)


def test_voice_response_sonification_uses_output_field_contrast() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    baseline_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
    )
    textured_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
    )
    textured_state = type(field.state)(
        phase=field.state.phase.copy(),
        frequency=field.state.frequency.copy(),
        metabolite=field.state.metabolite.copy(),
        coupling=field.state.coupling.copy(),
        trace=field.state.trace.copy(),
    )
    output_indices = np.argwhere(regions.output)
    alternating = np.arange(output_indices.shape[0]) % 2 == 0
    for index, (row, column) in enumerate(output_indices):
        if alternating[index]:
            textured_state.phase[row, column] += np.pi
            textured_state.trace[row, column] = 1.0
            textured_state.metabolite[row, column] = 0.35

    baseline = baseline_renderer.render_frame(
        field.state,
        regions,
        response_score=0.02,
    )
    textured = textured_renderer.render_frame(
        textured_state,
        regions,
        response_score=0.02,
    )

    assert not np.allclose(baseline, textured)
    assert np.all((-1.0 <= textured) & (textured <= 1.0))


def test_voice_response_sonification_uses_phase_texture_moments() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    bimodal_state = type(field.state)(
        phase=field.state.phase.copy(),
        frequency=field.state.frequency.copy(),
        metabolite=field.state.metabolite.copy(),
        coupling=field.state.coupling.copy(),
        trace=field.state.trace.copy(),
    )
    trimodal_state = type(field.state)(
        phase=field.state.phase.copy(),
        frequency=field.state.frequency.copy(),
        metabolite=field.state.metabolite.copy(),
        coupling=field.state.coupling.copy(),
        trace=field.state.trace.copy(),
    )
    output_indices = np.argwhere(regions.output)
    for index, (row, column) in enumerate(output_indices):
        bimodal_state.phase[row, column] = 0.0 if index % 2 == 0 else np.pi
        trimodal_state.phase[row, column] = (index % 3) * 2.0 * np.pi / 3.0

    bimodal_features = VoiceResponseSonificationRenderer._output_voice_features(
        bimodal_state,
        regions,
    )
    trimodal_features = VoiceResponseSonificationRenderer._output_voice_features(
        trimodal_state,
        regions,
    )
    bimodal_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
    )
    trimodal_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
    )

    bimodal = bimodal_renderer.render_frame(
        bimodal_state,
        regions,
        response_score=0.02,
    )
    trimodal = trimodal_renderer.render_frame(
        trimodal_state,
        regions,
        response_score=0.02,
    )

    assert bimodal_features["phase_order_2"] > trimodal_features["phase_order_2"]
    assert trimodal_features["phase_order_3"] > bimodal_features["phase_order_3"]
    assert not np.allclose(bimodal, trimodal)


def test_voice_response_sonification_keeps_short_response_memory() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    memory_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
        response_memory=0.6,
        response_memory_decay=0.01,
    )
    stateless_renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        smoothing=1.0,
        response_memory=0.0,
    )
    textured_state = type(field.state)(
        phase=field.state.phase.copy(),
        frequency=field.state.frequency.copy(),
        metabolite=field.state.metabolite.copy(),
        coupling=field.state.coupling.copy(),
        trace=field.state.trace.copy(),
    )
    textured_state.phase[regions.output] += np.pi / 2.0
    textured_state.trace[regions.output] = 1.0

    memory_renderer.render_frame(textured_state, regions, response_score=0.02)
    remembered = memory_renderer.render_frame(field.state, regions, response_score=0.02)
    stateless_renderer.render_frame(textured_state, regions, response_score=0.02)
    stateless = stateless_renderer.render_frame(
        field.state,
        regions,
        response_score=0.02,
    )

    assert not np.allclose(remembered, stateless)


def test_voice_response_sonification_articulates_response_events() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.5,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        release=0.5,
        background_level=0.0,
        background_response_level=0.0,
        articulation_attack=1.0,
        articulation_release=0.5,
        articulation_hold_frames=0,
        articulation_floor=0.0,
        energy_normalization_rate=1.0,
    )

    active = renderer.render_frame(field.state, regions, response_score=0.02)
    active_articulation = renderer.articulation
    fading = renderer.render_frame(field.state, regions, response_score=0.0)

    assert active_articulation > renderer.articulation
    assert np.sqrt(np.mean(active * active)) > np.sqrt(np.mean(fading * fading))


def test_voice_response_sonification_normalizes_response_energy() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    renderer = VoiceResponseSonificationRenderer(
        sample_rate=8_000,
        frame_size=128,
        gain=0.03,
        response_threshold=0.0,
        response_sensitivity=100.0,
        attack=1.0,
        background_level=0.0,
        background_response_level=0.0,
        articulation_attack=1.0,
        target_response_rms=0.08,
        energy_normalization_rate=1.0,
        max_energy_gain=3.0,
    )

    renderer.render_frame(field.state, regions, response_score=0.02)

    assert renderer.energy_gain > 1.0


def test_voice_response_sonification_rejects_invalid_response_threshold() -> None:
    with pytest.raises(ValueError, match="response_threshold"):
        VoiceResponseSonificationRenderer(response_threshold=-0.1)


def test_voice_response_sonification_rejects_invalid_background_levels() -> None:
    with pytest.raises(ValueError, match="background_level"):
        VoiceResponseSonificationRenderer(background_level=-0.1)
    with pytest.raises(ValueError, match="background_response_level"):
        VoiceResponseSonificationRenderer(
            background_level=0.2,
            background_response_level=0.1,
        )


def test_voice_response_sonification_rejects_invalid_articulation() -> None:
    with pytest.raises(ValueError, match="articulation_hold_frames"):
        VoiceResponseSonificationRenderer(articulation_hold_frames=-1)


def test_voice_response_sonification_rejects_invalid_energy_normalization() -> None:
    with pytest.raises(ValueError, match="target_response_rms"):
        VoiceResponseSonificationRenderer(target_response_rms=0.0)
    with pytest.raises(ValueError, match="max_energy_gain"):
        VoiceResponseSonificationRenderer(min_energy_gain=2.0, max_energy_gain=1.0)
