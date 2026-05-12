import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, RegionMasks, Simulation
from neuroacoustic_resonator.visualization import (
    LiveVisualizationConfig,
    frame_to_visualization,
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


def test_live_visualization_config_validates_values() -> None:
    with pytest.raises(ValueError, match="interval_ms"):
        LiveVisualizationConfig(interval_ms=0)
    with pytest.raises(ValueError, match="steps_per_update"):
        LiveVisualizationConfig(steps_per_update=0)
    with pytest.raises(ValueError, match="history_size"):
        LiveVisualizationConfig(history_size=1)
