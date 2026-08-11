from pathlib import Path

from neuroacoustic_resonator.analysis.distinguishability_corpus import (
    DistinguishabilityCorpusConfig,
)
from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationStimulus,
    build_calibration_trials,
)


def test_default_corpus_fixes_stimuli_seeds_and_repeats() -> None:
    corpus = DistinguishabilityCorpusConfig.from_file(
        "configs/distinguishability_corpus.yaml"
    )
    calibration = corpus.to_calibration_config()
    stimuli = [
        CalibrationStimulus(spec.label, Path(spec.label), source_type=spec.kind)
        for spec in calibration.synthetic_stimuli
    ]

    trials = build_calibration_trials(calibration, stimuli)

    assert {stimulus.kind for stimulus in corpus.stimuli} == {
        "silence",
        "tone",
        "chirp",
        "pulse",
        "noise",
    }
    assert corpus.seed_roots == (101, 211, 307, 401, 503, 601, 701)
    assert corpus.repeats == 2
    assert {item.seed_root: item.split for item in calibration.seed_splits} == {
        101: "train",
        211: "train",
        307: "train",
        401: "validation",
        503: "validation",
        601: "test",
        701: "test",
    }
    assert len(trials) == 70
    assert len({trial.trial_id for trial in trials}) == 70
    assert len({trial.field_seed for trial in trials}) == 14
    assert {trial.split for trial in trials} == {"train", "validation", "test"}
    assert all(
        left.field_seed != right.field_seed
        for left, right in zip(trials[::2], trials[1::2], strict=True)
    )
