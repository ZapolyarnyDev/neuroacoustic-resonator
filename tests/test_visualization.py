import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, RegionMasks, Simulation
from neuroacoustic_resonator.viz.live import (
    LiveVisualizationConfig,
    _LiveAudioOutput,
    frame_to_visualization,
    region_boundary_columns,
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
