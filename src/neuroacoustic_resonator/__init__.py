"""Core package for the neuroacoustic resonator."""

from neuroacoustic_resonator.audio.input import (
    AudioInputFeatures,
    WavInputDrive,
    extract_audio_input_features,
    write_audio_input_features_csv,
)
from neuroacoustic_resonator.audio.output import (
    ContinuousAudioRenderer,
    EventDrivenAudioRenderer,
    GatedAudioRenderer,
    SlopeTriggeredAudioRenderer,
    StimulusCoupledAudioRenderer,
    render_output_frame,
    write_wav,
)
from neuroacoustic_resonator.audio.render import render_audio_demo
from neuroacoustic_resonator.audio.conversation import (
    VoiceConversationConfig,
    render_voice_conversation,
)
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
from neuroacoustic_resonator.analysis.audio_input_run import (
    AudioInputRunConfig,
    run_audio_input_simulation,
)
from neuroacoustic_resonator.analysis.metrics import (
    MetricsHistory,
    RegionalActivityMetrics,
    RegionalActivityTracker,
    compute_regional_activity_metrics,
    region_activity,
    region_fast_activity,
    region_slow_activity,
)
from neuroacoustic_resonator.analysis.output_patterns import (
    OutputPatternHistory,
    OutputPatternSignature,
    compare_output_patterns,
    output_pattern_signature,
)
from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationStimulus,
    PatternCalibrationConfig,
    SyntheticStimulusSpec,
    run_pattern_calibration,
)
from neuroacoustic_resonator.analysis.reinforcement import (
    PatternReinforcementSignals,
    PatternReinforcementWeights,
    compute_pattern_reinforcement_signals,
)
from neuroacoustic_resonator.analysis.diagnostics_export import (
    export_diagnostics_artifacts,
)
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
    "AudioInputFeatures",
    "AudioInputRunConfig",
    "FieldConfig",
    "FieldConfigModel",
    "FieldMetrics",
    "FieldState",
    "ContinuousAudioRenderer",
    "CalibrationStimulus",
    "CheckpointPaths",
    "EventDrivenAudioRenderer",
    "GatedAudioRenderer",
    "LiveVisualizationConfig",
    "MetricsHistory",
    "OscillatorField",
    "OutputPatternHistory",
    "OutputPatternSignature",
    "PatternCalibrationConfig",
    "PatternReinforcementSignals",
    "PatternReinforcementWeights",
    "RegionMasks",
    "RealtimeAudioConfig",
    "RealtimeAudioEngine",
    "RegionalActivityMetrics",
    "RegionalActivityTracker",
    "Simulation",
    "SimulationConfig",
    "SimulationFrame",
    "SlopeTriggeredAudioRenderer",
    "StimulusCoupledAudioRenderer",
    "SyntheticInputConfig",
    "SyntheticInputDrive",
    "SyntheticStimulusSpec",
    "VisualizationFrame",
    "VoiceConversationConfig",
    "WavInputDrive",
    "compare_output_patterns",
    "compute_pattern_reinforcement_signals",
    "compute_regional_activity_metrics",
    "export_diagnostics_artifacts",
    "extract_audio_input_features",
    "frame_to_visualization",
    "checkpoint_paths",
    "load_field_state",
    "load_simulation_checkpoint",
    "region_activity",
    "region_fast_activity",
    "region_slow_activity",
    "output_pattern_signature",
    "render_audio_demo",
    "render_voice_conversation",
    "render_output_frame",
    "play_realtime_audio",
    "run_audio_input_simulation",
    "run_pattern_calibration",
    "save_field_state",
    "save_field_preview",
    "save_phase_preview",
    "save_simulation_checkpoint",
    "write_wav",
    "write_audio_input_features_csv",
]
