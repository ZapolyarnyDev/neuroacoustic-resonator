"""Core package for the neuroacoustic resonator."""

from neuroacoustic_resonator.analysis.audio_input_run import (
    AudioInputRunConfig,
    run_audio_input_simulation,
)
from neuroacoustic_resonator.analysis.diagnostics_export import (
    export_diagnostics_artifacts,
)
from neuroacoustic_resonator.analysis.metrics import (
    MetricsHistory,
    ProtocolActivityTracker,
    RegionalActivityMetrics,
    RegionalActivityTracker,
    compute_protocol_activity_metrics,
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
    pattern_features,
    protocol_pattern_signature,
)
from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationStimulus,
    PatternCalibrationConfig,
    SyntheticStimulusSpec,
    run_pattern_calibration,
)
from neuroacoustic_resonator.analysis.pattern_plasticity import (
    PatternGuidedPlasticityConfig,
    PatternPlasticityDecision,
    pattern_guided_plasticity_decision,
    summarize_plasticity_decisions,
)
from neuroacoustic_resonator.analysis.protocol_stream import (
    ProtocolAnalysisStream,
    ProtocolFrameHistory,
    ProtocolObservation,
    protocol_frame_row,
)
from neuroacoustic_resonator.analysis.reinforcement import (
    PatternReinforcementSignals,
    PatternReinforcementWeights,
    compute_pattern_reinforcement_signals,
)
from neuroacoustic_resonator.audio.conversation import (
    VoiceConversationConfig,
    render_voice_conversation,
)
from neuroacoustic_resonator.audio.input import (
    AudioInputFeatures,
    WavInputDrive,
    extract_audio_input_features,
    write_audio_input_features_csv,
)
from neuroacoustic_resonator.audio.io import write_wav
from neuroacoustic_resonator.configuration import FieldConfigModel, SimulationConfig
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
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame
from neuroacoustic_resonator.io.persistence import (
    CheckpointPaths,
    checkpoint_paths,
    load_field_state,
    load_simulation_checkpoint,
    save_field_state,
    save_simulation_checkpoint,
)
from neuroacoustic_resonator.viz.preview import save_field_preview, save_phase_preview

__all__ = [
    "AudioInputFeatures",
    "AudioInputRunConfig",
    "CalibrationStimulus",
    "CheckpointPaths",
    "FieldConfig",
    "FieldConfigModel",
    "FieldMetrics",
    "FieldState",
    "MetricsHistory",
    "OscillatorField",
    "OutputPatternHistory",
    "OutputPatternSignature",
    "PatternCalibrationConfig",
    "PatternGuidedPlasticityConfig",
    "PatternPlasticityDecision",
    "PatternReinforcementSignals",
    "PatternReinforcementWeights",
    "ProtocolActivityTracker",
    "ProtocolAnalysisStream",
    "ProtocolFrameHistory",
    "ProtocolObservation",
    "RegionMasks",
    "RegionalActivityMetrics",
    "RegionalActivityTracker",
    "Simulation",
    "SimulationConfig",
    "SimulationFrame",
    "SyntheticInputConfig",
    "SyntheticInputDrive",
    "SyntheticStimulusSpec",
    "VoiceConversationConfig",
    "WavInputDrive",
    "checkpoint_paths",
    "compare_output_patterns",
    "compute_pattern_reinforcement_signals",
    "compute_protocol_activity_metrics",
    "compute_regional_activity_metrics",
    "export_diagnostics_artifacts",
    "extract_audio_input_features",
    "load_field_state",
    "load_simulation_checkpoint",
    "output_pattern_signature",
    "pattern_features",
    "pattern_guided_plasticity_decision",
    "protocol_frame_row",
    "protocol_pattern_signature",
    "region_activity",
    "region_fast_activity",
    "region_slow_activity",
    "render_voice_conversation",
    "run_audio_input_simulation",
    "run_pattern_calibration",
    "save_field_preview",
    "save_field_state",
    "save_phase_preview",
    "save_simulation_checkpoint",
    "summarize_plasticity_decisions",
    "write_audio_input_features_csv",
    "write_wav",
]
