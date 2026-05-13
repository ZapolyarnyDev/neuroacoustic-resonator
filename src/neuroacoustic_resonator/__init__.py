"""Core package for the neuroacoustic resonator."""

from neuroacoustic_resonator.audio.output import (
    ContinuousAudioRenderer,
    EventDrivenAudioRenderer,
    GatedAudioRenderer,
    render_output_frame,
    write_wav,
)
from neuroacoustic_resonator.audio.render import render_audio_demo
from neuroacoustic_resonator.core.config import FieldConfigModel, SimulationConfig
from neuroacoustic_resonator.core.field import (
    FieldConfig,
    FieldMetrics,
    FieldState,
    OscillatorField,
)
from neuroacoustic_resonator.core.input_drive import (
    SyntheticInputConfig,
    SyntheticInputDrive,
)
from neuroacoustic_resonator.analysis.metrics import MetricsHistory
from neuroacoustic_resonator.io.persistence import (
    CheckpointPaths,
    checkpoint_paths,
    load_field_state,
    load_simulation_checkpoint,
    save_field_state,
    save_simulation_checkpoint,
)
from neuroacoustic_resonator.viz.preview import save_field_preview, save_phase_preview
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.audio.realtime import (
    RealtimeAudioConfig,
    RealtimeAudioEngine,
    play_realtime_audio,
)
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame
from neuroacoustic_resonator.viz.live import (
    LiveVisualizationConfig,
    VisualizationFrame,
    frame_to_visualization,
)

__all__ = [
    "FieldConfig",
    "FieldConfigModel",
    "FieldMetrics",
    "FieldState",
    "ContinuousAudioRenderer",
    "CheckpointPaths",
    "EventDrivenAudioRenderer",
    "GatedAudioRenderer",
    "LiveVisualizationConfig",
    "MetricsHistory",
    "OscillatorField",
    "RegionMasks",
    "RealtimeAudioConfig",
    "RealtimeAudioEngine",
    "Simulation",
    "SimulationConfig",
    "SimulationFrame",
    "SyntheticInputConfig",
    "SyntheticInputDrive",
    "VisualizationFrame",
    "frame_to_visualization",
    "checkpoint_paths",
    "load_field_state",
    "load_simulation_checkpoint",
    "render_audio_demo",
    "render_output_frame",
    "play_realtime_audio",
    "save_field_state",
    "save_field_preview",
    "save_phase_preview",
    "save_simulation_checkpoint",
    "write_wav",
]
