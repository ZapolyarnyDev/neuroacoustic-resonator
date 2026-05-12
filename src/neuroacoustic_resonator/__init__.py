"""Core package for the neuroacoustic resonator."""

from neuroacoustic_resonator.audio_output import render_output_frame, write_wav
from neuroacoustic_resonator.audio_render import render_audio_demo
from neuroacoustic_resonator.config import FieldConfigModel, SimulationConfig
from neuroacoustic_resonator.field import (
    FieldConfig,
    FieldMetrics,
    FieldState,
    OscillatorField,
)
from neuroacoustic_resonator.metrics import MetricsHistory
from neuroacoustic_resonator.preview import save_field_preview, save_phase_preview
from neuroacoustic_resonator.regions import RegionMasks
from neuroacoustic_resonator.simulation import Simulation, SimulationFrame
from neuroacoustic_resonator.visualization import (
    LiveVisualizationConfig,
    VisualizationFrame,
    frame_to_visualization,
)

__all__ = [
    "FieldConfig",
    "FieldConfigModel",
    "FieldMetrics",
    "FieldState",
    "LiveVisualizationConfig",
    "MetricsHistory",
    "OscillatorField",
    "RegionMasks",
    "Simulation",
    "SimulationConfig",
    "SimulationFrame",
    "VisualizationFrame",
    "frame_to_visualization",
    "render_audio_demo",
    "render_output_frame",
    "save_field_preview",
    "save_phase_preview",
    "write_wav",
]
