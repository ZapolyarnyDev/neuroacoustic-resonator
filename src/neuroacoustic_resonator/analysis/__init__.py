from typing import TYPE_CHECKING

from neuroacoustic_resonator.analysis.benchmark import (
    BenchmarkResult,
    benchmark_field_step,
    benchmark_sizes,
    write_benchmark_results,
)
from neuroacoustic_resonator.analysis.audio_input_run import (
    AudioInputRunConfig,
    run_audio_input_simulation,
)
from neuroacoustic_resonator.analysis.diagnostics_export import (
    export_diagnostics_artifacts,
)
from neuroacoustic_resonator.analysis.experiments import (
    ExperimentAnalysisConfig,
    ExperimentRows,
    ExperimentSummary,
    run_experiment_suite,
)
from neuroacoustic_resonator.analysis.metrics import (
    MetricsHistory,
    RegionalActivityMetrics,
    RegionalActivityTracker,
    compute_regional_activity_metrics,
    region_fast_activity,
    region_activity,
    region_slow_activity,
)
from neuroacoustic_resonator.analysis.output_patterns import (
    OutputPatternHistory,
    OutputPatternSignature,
    compare_output_patterns,
    output_pattern_signature,
)
from neuroacoustic_resonator.analysis.pattern_plasticity import (
    PatternGuidedPlasticityConfig,
    PatternPlasticityDecision,
    pattern_guided_plasticity_decision,
    summarize_plasticity_decisions,
)
from neuroacoustic_resonator.analysis.propagation_probe import (
    PropagationProbeConfig,
    run_propagation_probe,
)
from neuroacoustic_resonator.analysis.reinforcement import (
    PatternReinforcementSignals,
    PatternReinforcementWeights,
    compute_pattern_reinforcement_signals,
)
from neuroacoustic_resonator.analysis.voice_probe import (
    VoiceVsSilenceProbeConfig,
    run_voice_vs_silence_probe,
)
from neuroacoustic_resonator.analysis.voice_memory_probe import (
    VoiceMemoryProbeConfig,
    run_voice_memory_probe,
)

if TYPE_CHECKING:
    from neuroacoustic_resonator.analysis.pattern_calibration import (
        CalibrationStimulus,
        PatternCalibrationConfig,
        SyntheticStimulusSpec,
        run_pattern_calibration,
    )

__all__ = [
    "AudioInputRunConfig",
    "BenchmarkResult",
    "ExperimentAnalysisConfig",
    "ExperimentRows",
    "ExperimentSummary",
    "MetricsHistory",
    "OutputPatternHistory",
    "OutputPatternSignature",
    "PatternCalibrationConfig",
    "PatternGuidedPlasticityConfig",
    "PatternPlasticityDecision",
    "PatternReinforcementSignals",
    "PatternReinforcementWeights",
    "PropagationProbeConfig",
    "RegionalActivityMetrics",
    "RegionalActivityTracker",
    "CalibrationStimulus",
    "SyntheticStimulusSpec",
    "VoiceVsSilenceProbeConfig",
    "VoiceMemoryProbeConfig",
    "benchmark_field_step",
    "benchmark_sizes",
    "compare_output_patterns",
    "compute_pattern_reinforcement_signals",
    "compute_regional_activity_metrics",
    "export_diagnostics_artifacts",
    "region_activity",
    "region_fast_activity",
    "region_slow_activity",
    "output_pattern_signature",
    "pattern_guided_plasticity_decision",
    "run_audio_input_simulation",
    "run_pattern_calibration",
    "run_propagation_probe",
    "run_voice_vs_silence_probe",
    "run_voice_memory_probe",
    "run_experiment_suite",
    "summarize_plasticity_decisions",
    "write_benchmark_results",
]


def __getattr__(name: str) -> object:
    if name in {
        "CalibrationStimulus",
        "PatternCalibrationConfig",
        "SyntheticStimulusSpec",
        "run_pattern_calibration",
    }:
        from neuroacoustic_resonator.analysis import pattern_calibration

        return getattr(pattern_calibration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
