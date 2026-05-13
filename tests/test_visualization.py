import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, RegionMasks, Simulation
from neuroacoustic_resonator.audio.input import AudioInputFeatures, WavInputDrive
from neuroacoustic_resonator.viz.live import (
    DiagnosticsSnapshotRecorder,
    LiveVisualizationConfig,
    _LiveAudioOutput,
    diagnostic_curve_specs,
    diagnostic_legend_html,
    diagnostics_row,
    frame_to_visualization,
    region_boundary_columns,
    step_simulation_with_wav_input,
)


def test_frame_to_visualization_collects_maps_and_metrics() -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    frame = simulation.step()
    regions = RegionMasks.from_size(5)

    view_frame = frame_to_visualization(frame, regions)

    assert view_frame.phase.shape == (5, 5)
    assert view_frame.local_synchrony.shape == (5, 5)
    assert view_frame.metabolite.shape == (5, 5)
    assert view_frame.trace.shape == (5, 5)
    assert view_frame.region_labels.shape == (5, 5)
    assert view_frame.step == 1
    assert 0.0 <= view_frame.global_synchrony <= 1.0
    assert 0.0 <= view_frame.mean_metabolite <= 1.0
    assert view_frame.regional_metrics.output_activity >= 0.0
    assert view_frame.regional_metrics.output_event_score == 0.0
    assert set(np.unique(view_frame.region_labels)) == {1, 2, 3}


def test_frame_to_visualization_rejects_shape_mismatch() -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    frame = simulation.step()
    regions = RegionMasks.from_size(6)

    with pytest.raises(ValueError, match="matching shapes"):
        frame_to_visualization(frame, regions)


def test_region_boundary_columns_report_assoc_and_output_starts() -> None:
    regions = RegionMasks.from_size(10, edge_fraction=0.2)

    assert region_boundary_columns(regions) == (2, 8)


def test_live_visualization_config_validates_values() -> None:
    with pytest.raises(ValueError, match="interval_ms"):
        LiveVisualizationConfig(interval_ms=0)
    with pytest.raises(ValueError, match="steps_per_update"):
        LiveVisualizationConfig(steps_per_update=0)
    with pytest.raises(ValueError, match="history_size"):
        LiveVisualizationConfig(history_size=1)
    with pytest.raises(ValueError, match="audio_mode"):
        LiveVisualizationConfig(audio_mode="unknown")
    with pytest.raises(ValueError, match="audio_sample_rate"):
        LiveVisualizationConfig(audio_sample_rate=0)
    with pytest.raises(ValueError, match="diagnostics_sample_interval"):
        LiveVisualizationConfig(diagnostics_sample_interval=0)
    with pytest.raises(ValueError, match="audio_coupled_response_window_frames"):
        LiveVisualizationConfig(audio_coupled_response_window_frames=0)
    with pytest.raises(ValueError, match="audio_coupled_response_sensitivity"):
        LiveVisualizationConfig(audio_coupled_response_sensitivity=0.0)
    with pytest.raises(ValueError, match="input_frame_size"):
        LiveVisualizationConfig(input_frame_size=0)
    with pytest.raises(ValueError, match="input_drive_strength"):
        LiveVisualizationConfig(input_drive_strength=-0.1)


def test_diagnostic_curve_specs_have_stable_labels() -> None:
    specs = diagnostic_curve_specs()

    assert [spec.key for spec in specs] == [
        "global_synchrony",
        "mean_metabolite",
        "output_activity",
        "output_event_score",
        "output_fast_response_score",
        "output_slow_drift_score",
        "input_value",
        "audio_envelope",
        "stimulus_window",
        "coupled_audio_trigger",
    ]
    assert all(spec.label for spec in specs)
    assert all(len(spec.color) == 3 for spec in specs)


def test_diagnostic_legend_html_contains_labels_and_colors() -> None:
    html = diagnostic_legend_html(diagnostic_curve_specs())

    assert "global synchrony" in html
    assert "audio envelope" in html
    assert "rgb(" in html


def test_diagnostics_row_contains_plotted_values() -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    regions = RegionMasks.from_size(5)
    view_frame = frame_to_visualization(simulation.step(), regions)

    row = diagnostics_row(
        view_frame,
        audio_envelope=0.25,
        stimulus_window=0.5,
        coupled_audio_trigger=0.75,
    )

    assert row["step"] == 1
    assert row["audio_envelope"] == 0.25
    assert row["stimulus_window"] == 0.5
    assert row["coupled_audio_trigger"] == 0.75
    assert row["output_activity"] >= 0.0
    assert row["output_fast_activity"] >= 0.0
    assert row["output_slow_activity"] >= 0.0


def test_diagnostics_snapshot_recorder_writes_sampled_csv(tmp_path) -> None:
    output = tmp_path / "diagnostics.csv"
    recorder = DiagnosticsSnapshotRecorder(output, sample_interval=2)

    recorder.record(
        {
            "step": 1,
            "global_synchrony": 0.1,
            "mean_metabolite": 0.2,
            "output_activity": 0.3,
            "output_fast_activity": 0.31,
            "output_slow_activity": 0.32,
            "output_event_score": 0.4,
            "output_fast_response_score": 0.41,
            "output_slow_drift_score": 0.42,
            "input_value": 0.5,
            "audio_envelope": 0.6,
            "stimulus_window": 0.7,
            "coupled_audio_trigger": 0.8,
        },
        step=1,
    )
    recorder.record(
        {
            "step": 2,
            "global_synchrony": 0.2,
            "mean_metabolite": 0.3,
            "output_activity": 0.4,
            "output_fast_activity": 0.41,
            "output_slow_activity": 0.42,
            "output_event_score": 0.5,
            "output_fast_response_score": 0.51,
            "output_slow_drift_score": 0.52,
            "input_value": 0.6,
            "audio_envelope": 0.7,
            "stimulus_window": 0.8,
            "coupled_audio_trigger": 0.9,
        },
        step=2,
    )

    assert "step,global_synchrony" in output.read_text(encoding="utf-8")
    assert "2,0.2,0.3,0.4,0.41,0.42,0.5,0.51,0.52,0.6,0.7,0.8,0.9" in output.read_text(
        encoding="utf-8"
    )


def test_live_audio_output_callback_uses_latest_state() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    audio_output = _LiveAudioOutput(
        config=LiveVisualizationConfig(
            audio_enabled=True,
            audio_sample_rate=8_000,
            audio_frame_size=32,
            audio_mode="continuous",
        ),
        regions=regions,
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    audio_output.callback(outdata, 32, None, None)
    assert np.max(np.abs(outdata)) == 0.0

    audio_output.update_state(simulation.step().state)
    audio_output.callback(outdata, 32, None, None)

    assert np.all(np.isfinite(outdata))
    assert np.max(np.abs(outdata)) > 0.0
    assert audio_output.envelope == 0.0


def test_step_simulation_with_wav_input_routes_drive_to_input_region() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    features = AudioInputFeatures(
        sample_rate=8_000,
        frame_size=64,
        hop_size=64,
        rms=np.asarray([0.0, 1.0]),
        onset=np.asarray([0.0, 1.0]),
        spectral_centroid=np.asarray([0.0, 0.5]),
        drive=np.asarray([0.0, 0.25]),
    )
    drive = WavInputDrive(features, regions)
    before = simulation.field.state.phase.copy()

    frame = step_simulation_with_wav_input(simulation, drive, input_step=1)

    assert simulation.step_index == 1
    assert simulation.last_input_value == 0.25
    assert frame.metrics.step == 1
    assert not np.allclose(
        simulation.field.state.phase[regions.input], before[regions.input]
    )


def test_live_audio_output_supports_event_mode() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    audio_output = _LiveAudioOutput(
        config=LiveVisualizationConfig(
            audio_enabled=True,
            audio_sample_rate=8_000,
            audio_frame_size=32,
            audio_mode="event",
        ),
        regions=regions,
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    audio_output.update_state(simulation.step().state)
    audio_output.callback(outdata, 32, None, None)

    assert np.all(np.isfinite(outdata))
    assert audio_output.envelope >= 0.0


def test_live_audio_output_supports_slope_mode() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    audio_output = _LiveAudioOutput(
        config=LiveVisualizationConfig(
            audio_enabled=True,
            audio_sample_rate=8_000,
            audio_frame_size=32,
            audio_mode="slope",
        ),
        regions=regions,
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    audio_output.update_state(simulation.step().state)
    audio_output.callback(outdata, 32, None, None)

    assert np.all(np.isfinite(outdata))
    assert audio_output.envelope >= 0.0


def test_live_audio_output_supports_coupled_mode() -> None:
    simulation = Simulation(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    audio_output = _LiveAudioOutput(
        config=LiveVisualizationConfig(
            audio_enabled=True,
            audio_sample_rate=8_000,
            audio_frame_size=32,
            audio_mode="coupled",
        ),
        regions=regions,
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    audio_output.update_state(simulation.step().state, input_value=0.5)
    audio_output.callback(outdata, 32, None, None)

    assert np.all(np.isfinite(outdata))
    assert audio_output.stimulus_window >= 0.0
    assert audio_output.coupled_audio_trigger >= 0.0


def test_live_audio_output_status_reporting_is_throttled(capsys) -> None:
    regions = RegionMasks.from_size(6)
    audio_output = _LiveAudioOutput(
        config=LiveVisualizationConfig(
            audio_enabled=True,
            audio_sample_rate=8_000,
            audio_frame_size=32,
        ),
        regions=regions,
    )

    audio_output._report_status("output underflow")
    audio_output._report_status("output underflow")
    audio_output._report_status("different status")

    captured = capsys.readouterr()
    assert captured.out.count("output underflow") == 1
    assert captured.out.count("different status") == 1
