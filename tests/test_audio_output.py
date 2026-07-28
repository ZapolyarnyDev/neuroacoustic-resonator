from __future__ import annotations

import wave
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.audio.io import write_wav
from neuroacoustic_resonator.audio.output import ProtocolReferenceRenderer
from neuroacoustic_resonator.audio.synthesis import PatternVoiceSynthesizer
from neuroacoustic_resonator.core.simulation import SimulationFrame
from neuroacoustic_resonator.encoding import ProtocolEncoder
from neuroacoustic_resonator.protocol import (
    ActivePattern,
    FieldSnapshot,
    PatternSnapshot,
    PatternTransition,
    ProtocolConsumer,
    ProtocolReplay,
    RegionSnapshot,
    SoundProtocolFrame,
    write_protocol_jsonl,
)


def _region(name: str) -> RegionSnapshot:
    return RegionSnapshot(
        name=name,  # type: ignore[arg-type]
        phase_coherence=0.65,
        mean_local_synchrony=0.6,
        mean_metabolite=0.8,
        min_metabolite=0.5,
        mean_trace=0.3,
        max_trace=0.7,
        mean_frequency=1.0,
        frequency_spread=0.2,
        mean_coupling=0.25,
        coupling_spread=0.05,
    )


def _pattern(**changes: float) -> PatternSnapshot:
    values = {
        "phase_order_1": 0.65,
        "phase_order_2": 0.3,
        "phase_order_3": 0.2,
        "trace_mean": 0.3,
        "trace_contrast": 0.2,
        "metabolite_stress": 0.2,
        "metabolite_contrast": 0.15,
        "trace_phase_lock": 0.4,
        "metabolite_phase_lock": 0.35,
        "frequency_mean": 1.0,
        "frequency_spread": 0.2,
    }
    values.update(changes)
    return PatternSnapshot(**values)


def _frame(
    *,
    sequence: int = 0,
    pattern: PatternSnapshot | None = None,
    active_pattern: ActivePattern | None = None,
    transition: PatternTransition | None = None,
) -> SoundProtocolFrame:
    return SoundProtocolFrame(
        sequence=sequence,
        step=sequence,
        time_seconds=sequence * 0.05,
        field=FieldSnapshot(
            global_synchrony=0.5,
            mean_local_synchrony=0.55,
            mean_metabolite=0.75,
            min_metabolite=0.45,
            mean_trace=0.25,
            max_trace=0.7,
        ),
        input_region=_region("input"),
        assoc_region=_region("assoc"),
        output_region=_region("output"),
        pattern=pattern or _pattern(),
        active_pattern=active_pattern,
        transition=transition,
    )


def _active(
    *,
    label: str = "coherent",
    confidence: float = 1.0,
    intensity: float = 1.0,
    novelty: float = 0.0,
    is_novel: bool = False,
    age_frames: int = 1,
) -> ActivePattern:
    return ActivePattern(
        label=label,
        confidence=confidence,
        intensity=intensity,
        novelty=novelty,
        is_novel=is_novel,
        age_frames=age_frames,
    )


def _renderer(**changes: Any) -> ProtocolReferenceRenderer:
    options: dict[str, Any] = {
        "sample_rate": 8_000,
        "frame_size": 128,
        "gain": 0.5,
        "response_threshold": 0.0,
        "response_sensitivity": 100.0,
        "attack": 1.0,
        "smoothing": 1.0,
    }
    options.update(changes)
    return ProtocolReferenceRenderer(**options)


def test_write_wav_creates_pcm_file(tmp_path) -> None:
    audio = np.linspace(-1.0, 1.0, 16)

    output_path = write_wav(tmp_path / "demo.wav", audio, sample_rate=8_000)

    assert output_path.exists()
    with wave.open(str(output_path), "rb") as stream:
        assert stream.getframerate() == 8_000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() == 16


def test_reference_renderer_is_a_protocol_consumer() -> None:
    renderer = _renderer()
    frame = _frame(active_pattern=_active())

    renderer.consume(frame)

    assert isinstance(renderer, ProtocolConsumer)
    assert renderer.last_audio_frame.shape == (128,)
    assert np.max(np.abs(renderer.last_audio_frame)) > 0.0


def test_reference_renderer_activates_from_protocol_state() -> None:
    renderer = _renderer(frame_size=64)
    quiet = renderer.render_frame(
        _frame(
            pattern=_pattern(
                phase_order_1=0.0,
                trace_mean=0.0,
                metabolite_stress=0.0,
            )
        )
    )
    active = renderer.render_frame(
        _frame(
            sequence=1,
            active_pattern=_active(),
            transition=PatternTransition(
                kind="started",
                from_label=None,
                to_label="coherent",
            ),
        )
    )

    assert renderer.last_activation > 0.0
    assert renderer.envelope > 0.0
    assert np.max(np.abs(active)) > np.max(np.abs(quiet))
    assert np.all((active >= -1.0) & (active <= 1.0))


def test_reference_renderer_softens_activation_before_ceiling() -> None:
    renderer = _renderer(frame_size=64, response_sensitivity=220.0)

    renderer.render_frame(_frame(active_pattern=_active(intensity=0.05)))
    moderate_activation = renderer.last_activation
    renderer.render_frame(
        _frame(sequence=1, active_pattern=_active(intensity=0.5, age_frames=2))
    )

    assert 0.0 < moderate_activation < 0.5
    assert moderate_activation < renderer.last_activation < 1.0


def test_reference_renderer_keeps_idle_background_quiet() -> None:
    renderer = _renderer(
        frame_size=64,
        background_level=0.02,
        background_response_level=0.25,
    )

    quiet = renderer.render_frame(_frame())
    active = renderer.render_frame(_frame(sequence=1, active_pattern=_active()))

    assert np.max(np.abs(quiet)) < 0.02
    assert np.max(np.abs(active)) > np.max(np.abs(quiet)) * 4.0


def test_reference_renderer_changes_with_protocol_snapshot() -> None:
    renderer = _renderer()
    active = _active()
    renderer.render_frame(_frame(active_pattern=active))
    before = renderer.render_frame(
        _frame(sequence=1, active_pattern=replace(active, age_frames=2))
    )
    changed = _frame(
        sequence=2,
        pattern=_pattern(
            trace_mean=0.8,
            trace_contrast=0.6,
            phase_order_1=0.2,
        ),
        active_pattern=replace(active, age_frames=3),
    )

    after = renderer.render_frame(changed)

    assert not np.allclose(before, after)


def test_reference_renderer_uses_phase_texture_moments() -> None:
    active = _active(label="split")
    bimodal_frame = _frame(
        pattern=_pattern(phase_order_2=0.95, phase_order_3=0.1),
        active_pattern=active,
    )
    trimodal_frame = _frame(
        pattern=_pattern(phase_order_2=0.1, phase_order_3=0.95),
        active_pattern=replace(active, label="clustered"),
    )
    bimodal_features = ProtocolReferenceRenderer._protocol_voice_features(bimodal_frame)
    trimodal_features = ProtocolReferenceRenderer._protocol_voice_features(
        trimodal_frame
    )

    bimodal = _renderer().render_frame(bimodal_frame)
    trimodal = _renderer().render_frame(trimodal_frame)

    assert bimodal_features["phase_order_2"] > trimodal_features["phase_order_2"]
    assert trimodal_features["phase_order_3"] > bimodal_features["phase_order_3"]
    assert not np.allclose(bimodal, trimodal)


def test_reference_renderer_computes_acoustic_controls_inside_consumer() -> None:
    frame = _frame(
        pattern=_pattern(
            phase_order_1=0.4,
            phase_order_2=0.8,
            phase_order_3=0.2,
            trace_contrast=0.3,
            metabolite_stress=0.25,
            metabolite_contrast=0.2,
            frequency_mean=1.4,
            frequency_spread=0.35,
        )
    )

    features = ProtocolReferenceRenderer._protocol_voice_features(frame)

    assert features["brightness"] == pytest.approx(0.43)
    assert features["roughness"] == pytest.approx(0.435)
    assert not hasattr(frame.pattern, "brightness")
    assert not hasattr(frame.output_region, "roughness")


def test_reference_renderer_tracks_protocol_pattern_label() -> None:
    renderer = _renderer()
    frame = _frame(active_pattern=_active(label="split", confidence=0.85))

    audio = renderer.render_frame(frame)

    assert renderer.last_pattern_label == "split"
    assert renderer.last_pattern_confidence == pytest.approx(0.85)
    assert np.max(np.abs(audio)) > 0.0


def test_pattern_voice_depth_changes_reference_timbre() -> None:
    frame = _frame(active_pattern=_active(label="split", confidence=0.9))
    plain = _renderer(pattern_voice_depth=0.0)
    patterned = _renderer(pattern_voice_depth=1.0)

    plain_audio = plain.render_frame(frame)
    patterned_audio = patterned.render_frame(frame)

    assert not np.allclose(plain_audio, patterned_audio)


def test_pattern_voice_synthesizer_separates_pattern_waveforms() -> None:
    features = ProtocolReferenceRenderer._empty_voice_features()
    features.update(
        {
            "synchrony": 0.4,
            "trace": 0.7,
            "phase_order_2": 0.8,
            "phase_order_3": 0.2,
            "phase_spread": 0.6,
            "brightness": 0.5,
            "roughness": 0.4,
        }
    )
    split = PatternVoiceSynthesizer(
        sample_rate=8_000,
        frame_size=256,
        gain=0.5,
        smoothing=1.0,
        pattern_voice_depth=1.2,
    )
    diffuse = PatternVoiceSynthesizer(
        sample_rate=8_000,
        frame_size=256,
        gain=0.5,
        smoothing=1.0,
        pattern_voice_depth=1.2,
    )

    split_audio = split.render(
        features,
        pattern_label="split",
        pattern_confidence=0.9,
    )
    diffuse_audio = diffuse.render(
        features,
        pattern_label="diffuse",
        pattern_confidence=0.9,
    )

    assert not np.allclose(split_audio, diffuse_audio)
    assert abs(float(np.mean(split_audio)) - float(np.mean(diffuse_audio))) > 1e-4


def test_reference_renderer_keeps_short_response_memory() -> None:
    memory_renderer = _renderer(response_memory=0.6, response_memory_decay=0.01)
    stateless_renderer = _renderer(response_memory=0.0)
    textured = _frame(
        pattern=_pattern(trace_mean=0.9, phase_order_2=0.9),
        active_pattern=_active(label="split"),
    )
    baseline = _frame(
        sequence=1,
        active_pattern=_active(age_frames=2),
    )

    memory_renderer.render_frame(textured)
    remembered = memory_renderer.render_frame(baseline)
    stateless_renderer.render_frame(textured)
    stateless = stateless_renderer.render_frame(baseline)

    assert not np.allclose(remembered, stateless)


def test_reference_renderer_articulates_protocol_events() -> None:
    renderer = _renderer(
        release=0.5,
        background_level=0.0,
        background_response_level=0.0,
        articulation_attack=1.0,
        articulation_release=0.5,
        articulation_hold_frames=0,
        articulation_floor=0.0,
        energy_normalization_rate=1.0,
    )

    active = renderer.render_frame(
        _frame(
            active_pattern=_active(),
            transition=PatternTransition(
                kind="started",
                from_label=None,
                to_label="coherent",
            ),
        )
    )
    active_articulation = renderer.articulation
    fading = renderer.render_frame(
        _frame(
            sequence=1,
            pattern=_pattern(
                phase_order_1=0.0,
                trace_mean=0.0,
                metabolite_stress=0.0,
            ),
        )
    )

    assert active_articulation > renderer.articulation
    assert np.sqrt(np.mean(active * active)) > np.sqrt(np.mean(fading * fading))


def test_reference_renderer_min_response_gain_prevents_hard_fade() -> None:
    renderer = _renderer(
        release=0.8,
        background_level=0.0,
        background_response_level=0.0,
        articulation_attack=1.0,
        articulation_release=1.0,
        articulation_hold_frames=0,
        articulation_floor=0.0,
        min_response_gain=0.3,
        energy_normalization_rate=1.0,
    )

    active = renderer.render_frame(_frame(active_pattern=_active()))
    fading = renderer.render_frame(
        _frame(
            sequence=1,
            pattern=_pattern(
                phase_order_1=0.0,
                trace_mean=0.0,
                metabolite_stress=0.0,
            ),
        )
    )

    active_rms = float(np.sqrt(np.mean(active * active)))
    fading_rms = float(np.sqrt(np.mean(fading * fading)))
    assert fading_rms > active_rms * 0.12


def test_reference_renderer_normalizes_response_energy() -> None:
    renderer = _renderer(
        gain=0.03,
        background_level=0.0,
        background_response_level=0.0,
        articulation_attack=1.0,
        target_response_rms=0.08,
        energy_normalization_rate=1.0,
        max_energy_gain=3.0,
    )

    renderer.render_frame(_frame(active_pattern=_active()))

    assert renderer.energy_gain > 1.0


def test_live_and_replay_produce_identical_reference_audio(tmp_path) -> None:
    frames = [
        _frame(
            sequence=index,
            active_pattern=_active(age_frames=index + 1),
            transition=(
                PatternTransition(
                    kind="started",
                    from_label=None,
                    to_label="coherent",
                )
                if index == 0
                else None
            ),
        )
        for index in range(6)
    ]
    recording = write_protocol_jsonl(tmp_path / "audio.jsonl", frames)
    live_renderer = _renderer()
    replay_renderer = _renderer()

    live_audio = np.concatenate([live_renderer.render_frame(frame) for frame in frames])
    replay_audio = np.concatenate(
        [replay_renderer.render_frame(frame) for frame in ProtocolReplay(recording)]
    )

    np.testing.assert_allclose(replay_audio, live_audio, rtol=0.0, atol=0.0)


def test_protocol_reference_wav_stays_within_legacy_numeric_tolerance() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    simulation_frame = SimulationFrame(
        state=field.state,
        metrics=field.metrics(step=0),
        local_synchrony=field.local_synchrony(),
    )
    encoded = ProtocolEncoder(dt=field.config.dt).encode(simulation_frame, regions)
    assert encoded is not None
    active = _active()
    frames = [
        replace(
            encoded,
            sequence=index,
            step=index,
            time_seconds=index * field.config.dt,
            active_pattern=replace(active, age_frames=index + 1),
            transition=(
                PatternTransition(
                    kind="started",
                    from_label=None,
                    to_label=active.label,
                )
                if index == 0
                else None
            ),
        )
        for index in range(8)
    ]
    renderer = _renderer()

    audio = np.concatenate([renderer.render_frame(frame) for frame in frames])
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / 8_000)
    centroid = float(np.sum(frequencies * spectrum) / np.sum(spectrum))

    assert audio.size == 1_024
    assert rms == pytest.approx(0.2025525761, rel=0.1)
    assert peak == pytest.approx(0.4702342324, rel=0.1)
    assert centroid == pytest.approx(386.1511960, rel=0.1)


def test_reference_renderer_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="response_threshold"):
        ProtocolReferenceRenderer(response_threshold=-0.1)
    with pytest.raises(ValueError, match="background_level"):
        ProtocolReferenceRenderer(background_level=-0.1)
    with pytest.raises(ValueError, match="background_response_level"):
        ProtocolReferenceRenderer(
            background_level=0.2,
            background_response_level=0.1,
        )
    with pytest.raises(ValueError, match="articulation_hold_frames"):
        ProtocolReferenceRenderer(articulation_hold_frames=-1)
    with pytest.raises(ValueError, match="target_response_rms"):
        ProtocolReferenceRenderer(target_response_rms=0.0)
    with pytest.raises(ValueError, match="min_response_gain"):
        ProtocolReferenceRenderer(min_response_gain=-0.1)
    with pytest.raises(ValueError, match="max_energy_gain"):
        ProtocolReferenceRenderer(min_energy_gain=2.0, max_energy_gain=1.0)
    with pytest.raises(ValueError, match="pattern_voice_depth"):
        ProtocolReferenceRenderer(pattern_voice_depth=-0.1)
