from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.analysis.distinguishability_classification import (
    classify_protocol_embeddings,
)
from neuroacoustic_resonator.analysis.distinguishability_corpus import (
    DistinguishabilityCorpusConfig,
    run_seeded_distinguishability_corpus,
)
from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    quantify_response_separation,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    extract_protocol_embeddings,
)
from neuroacoustic_resonator.configuration import SimulationConfig


class ScenarioFieldOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coupling_strength: float | None = Field(default=None, ge=0.0)
    frequency_spread: float | None = Field(default=None, ge=0.0)


class ScenarioCorpusOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drive_strength: float | None = Field(default=None, ge=0.0)


class RobustnessScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    field: ScenarioFieldOverrides = Field(default_factory=ScenarioFieldOverrides)
    corpus: ScenarioCorpusOverrides = Field(default_factory=ScenarioCorpusOverrides)

    @model_validator(mode="after")
    def validate_overrides(self) -> Self:
        override_count = len(self.field.model_dump(exclude_none=True)) + len(
            self.corpus.model_dump(exclude_none=True)
        )
        if self.name == "baseline" and override_count != 0:
            msg = "baseline scenario must not override parameters"
            raise ValueError(msg)
        if self.name != "baseline" and override_count != 1:
            msg = "each robustness scenario must override exactly one parameter"
            raise ValueError(msg)
        return self


class EvidenceGateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_seed_roots: int = Field(default=7, ge=2)
    min_test_seed_roots: int = Field(default=2, ge=2)
    min_separation_margin_ci_low: float = 0.0
    min_cliffs_delta_ci_low: float = 0.0
    min_test_balanced_accuracy_ci_low: float = Field(default=0.35, ge=0.0, le=1.0)
    max_permutation_p_value: float = Field(default=0.05, gt=0.0, le=1.0)
    min_silence_recall: float = Field(default=0.5, ge=0.0, le=1.0)
    min_scenario_pass_rate: float = Field(default=1.0, gt=0.0, le=1.0)


class StageOneEvidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_config: Path
    output_root: Path
    output_report: Path
    scenarios: tuple[RobustnessScenarioConfig, ...] = Field(min_length=1)
    thresholds: EvidenceGateThresholds = Field(default_factory=EvidenceGateThresholds)
    bootstrap_samples: int = Field(default=5_000, ge=1)
    bootstrap_seed: int = Field(default=20_260_811, ge=0)
    permutation_samples: int = Field(default=2_000, ge=1)
    permutation_seed: int = Field(default=20_260_812, ge=0)

    @model_validator(mode="after")
    def validate_scenarios(self) -> Self:
        names = [scenario.name for scenario in self.scenarios]
        if len(set(names)) != len(names):
            msg = "robustness scenario names must be unique"
            raise ValueError(msg)
        if names.count("baseline") != 1:
            msg = "robustness matrix must contain exactly one baseline"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> StageOneEvidenceConfig:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls.model_validate(data)


@dataclass(frozen=True)
class PreparedScenario:
    name: str
    root: Path
    simulation_config: Path
    corpus_config: Path
    manifest: Path
    embeddings: Path
    embedding_schema: Path
    separation_report: Path
    distance_pairs: Path
    separation_plot: Path
    classification_report: Path
    predictions: Path
    confusion_plot: Path
    result: Path


def prepare_stage_one(config: StageOneEvidenceConfig) -> list[PreparedScenario]:
    base_corpus = DistinguishabilityCorpusConfig.from_file(config.corpus_config)
    base_simulation = SimulationConfig.from_file(base_corpus.simulation_config)
    prepared = []
    for scenario in config.scenarios:
        item = scenario_paths(config.output_root, scenario.name)
        item.root.mkdir(parents=True, exist_ok=True)
        simulation = apply_field_overrides(base_simulation, scenario)
        corpus = apply_corpus_overrides(base_corpus, scenario, item)
        write_yaml(item.simulation_config, simulation.model_dump(mode="json"))
        corpus = corpus.model_copy(update={"simulation_config": item.simulation_config})
        write_yaml(item.corpus_config, corpus.model_dump(mode="json"))
        write_json(
            item.root / "scenario.json",
            {
                "name": scenario.name,
                "field_overrides": scenario.field.model_dump(exclude_none=True),
                "corpus_overrides": scenario.corpus.model_dump(exclude_none=True),
                "simulation_config": str(item.simulation_config),
                "corpus_config": str(item.corpus_config),
                "result": str(item.result),
            },
        )
        prepared.append(item)
    return prepared


def apply_field_overrides(
    simulation: SimulationConfig,
    scenario: RobustnessScenarioConfig,
) -> SimulationConfig:
    overrides = scenario.field.model_dump(exclude_none=True)
    if not overrides:
        return simulation
    field = simulation.field.model_copy(update=overrides)
    return simulation.model_copy(update={"field": field})


def apply_corpus_overrides(
    corpus: DistinguishabilityCorpusConfig,
    scenario: RobustnessScenarioConfig,
    paths: PreparedScenario,
) -> DistinguishabilityCorpusConfig:
    overrides: dict[str, Any] = scenario.corpus.model_dump(exclude_none=True)
    overrides.update(
        {
            "output_dir": paths.root / "trials",
            "output_csv": paths.root / "calibration.csv",
            "output_summary": paths.root / "calibration_summary.json",
            "output_manifest": paths.manifest,
        }
    )
    return corpus.model_copy(update=overrides)


def scenario_paths(output_root: Path, name: str) -> PreparedScenario:
    root = output_root / name
    return PreparedScenario(
        name=name,
        root=root,
        simulation_config=root / "simulation.yaml",
        corpus_config=root / "corpus.yaml",
        manifest=root / "manifest.json",
        embeddings=root / "embeddings.csv",
        embedding_schema=root / "embedding_schema.json",
        separation_report=root / "separation.json",
        distance_pairs=root / "distance_pairs.csv",
        separation_plot=root / "separation.png",
        classification_report=root / "classification.json",
        predictions=root / "predictions.csv",
        confusion_plot=root / "confusion.png",
        result=root / "scenario_result.json",
    )


def run_prepared_scenario(
    config: StageOneEvidenceConfig,
    scenario: PreparedScenario,
    *,
    analysis_only: bool,
) -> dict[str, Any]:
    if not analysis_only:
        run_seeded_distinguishability_corpus(scenario.corpus_config)
    extract_protocol_embeddings(
        scenario.manifest,
        scenario.embeddings,
        scenario.embedding_schema,
    )
    separation = quantify_response_separation(
        scenario.embeddings,
        scenario.separation_report,
        scenario.distance_pairs,
        scenario.separation_plot,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
    )
    classification = classify_protocol_embeddings(
        scenario.embeddings,
        scenario.classification_report,
        scenario.predictions,
        scenario.confusion_plot,
        permutation_samples=config.permutation_samples,
        permutation_seed=config.permutation_seed,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed + 1,
    )
    result = build_scenario_result(
        scenario,
        separation,
        classification,
        config.thresholds,
    )
    write_json(scenario.result, result)
    return result


def build_scenario_result(
    scenario: PreparedScenario,
    separation: dict[str, Any],
    classification: dict[str, Any],
    thresholds: EvidenceGateThresholds,
) -> dict[str, Any]:
    gate = evaluate_scenario_gate(separation, classification, thresholds)
    return {
        "scenario": scenario.name,
        "status": "passed" if gate["passed"] else "failed",
        "gate": gate,
        "metrics": {
            "seed_roots": separation["overall"]["seed_roots"],
            "separation_margin": separation["overall"]["separation_margin"],
            "separation_margin_ci": separation["overall"]["separation_margin_ci"],
            "cliffs_delta": separation["overall"]["cliffs_delta"],
            "cliffs_delta_ci": separation["overall"]["cliffs_delta_ci"],
            "test_balanced_accuracy": classification["test"]["balanced_accuracy"],
            "test_balanced_accuracy_ci": classification["test"]["balanced_accuracy_ci"],
            "permutation_p_value": classification["permutation_baseline"]["p_value"],
            "silence_recall": classification["test"]["per_class_recall"].get("silence"),
        },
        "reports": {
            "separation": str(scenario.separation_report),
            "classification": str(scenario.classification_report),
        },
    }


def evaluate_scenario_gate(
    separation: dict[str, Any],
    classification: dict[str, Any],
    thresholds: EvidenceGateThresholds,
) -> dict[str, Any]:
    overall = require_available_separation(separation)
    test = classification["test"]
    baseline = classification["permutation_baseline"]
    seed_roots = overall["seed_roots"]
    test_seed_roots = test["seed_roots"]
    margin_ci_low = float(overall["separation_margin_ci"][0])
    cliffs_ci_low = float(overall["cliffs_delta_ci"][0])
    accuracy_ci_low = float(test["balanced_accuracy_ci"][0])
    test_accuracy = float(test["balanced_accuracy"])
    baseline_ci_high = float(baseline["balanced_accuracy_ci"][1])
    silence_recall = float(test["per_class_recall"].get("silence", 0.0))
    checks = {
        "enough_seed_roots": len(seed_roots) >= thresholds.min_seed_roots,
        "enough_test_seed_roots": (
            len(test_seed_roots) >= thresholds.min_test_seed_roots
        ),
        "separation_margin_ci": (
            margin_ci_low > thresholds.min_separation_margin_ci_low
        ),
        "cliffs_delta_ci": cliffs_ci_low > thresholds.min_cliffs_delta_ci_low,
        "test_accuracy_ci": (
            accuracy_ci_low >= thresholds.min_test_balanced_accuracy_ci_low
        ),
        "beats_permutation_interval": test_accuracy > baseline_ci_high,
        "permutation_p_value": (
            float(baseline["p_value"]) <= thresholds.max_permutation_p_value
        ),
        "silence_control": silence_recall >= thresholds.min_silence_recall,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds.model_dump(),
    }


def require_available_separation(report: dict[str, Any]) -> dict[str, Any]:
    overall = report.get("overall")
    if not isinstance(overall, dict) or not overall.get("available"):
        msg = "separation report has no available overall result"
        raise ValueError(msg)
    return overall


def evaluate_stage_one_gate(
    config: StageOneEvidenceConfig,
    scenarios: list[PreparedScenario],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for scenario in scenarios:
        if not scenario.result.exists():
            missing.append(scenario.name)
            continue
        value = json.loads(scenario.result.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            msg = f"scenario result must be an object: {scenario.result}"
            raise ValueError(msg)
        results.append(value)
    passed_count = sum(bool(result["gate"]["passed"]) for result in results)
    expected_count = len(scenarios)
    pass_rate = passed_count / expected_count
    baseline = next(
        (result for result in results if result["scenario"] == "baseline"),
        None,
    )
    checks = {
        "all_scenarios_complete": not missing,
        "baseline_passed": bool(baseline and baseline["gate"]["passed"]),
        "scenario_pass_rate": pass_rate >= config.thresholds.min_scenario_pass_rate,
    }
    passed = all(checks.values())
    status = "passed" if passed else "incomplete" if missing else "failed"
    report = {
        "stage": "distinguishability_and_repeatability",
        "status": status,
        "passed": passed,
        "checks": checks,
        "expected_scenarios": expected_count,
        "completed_scenarios": len(results),
        "passed_scenarios": passed_count,
        "scenario_pass_rate": pass_rate,
        "missing_scenarios": missing,
        "thresholds": config.thresholds.model_dump(),
        "scenarios": results,
    }
    write_json(config.output_report, report)
    return report


def write_yaml(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    return output


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 robustness matrix and evidence gate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "distinguishability_stage_one.yaml",
    )
    parser.add_argument("--scenario", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--gate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StageOneEvidenceConfig.from_file(args.config)
    scenarios = prepare_stage_one(config)
    if args.prepare_only:
        print(
            json.dumps(
                {"prepared": [scenario.name for scenario in scenarios]},
                indent=2,
            )
        )
        return 0
    selected_names = set(args.scenario)
    unknown = selected_names - {scenario.name for scenario in scenarios}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown robustness scenarios: {names}")
    if not args.gate_only:
        selected = [
            scenario
            for scenario in scenarios
            if not selected_names or scenario.name in selected_names
        ]
        selected_results = []
        for scenario in selected:
            selected_results.append(
                run_prepared_scenario(
                    config,
                    scenario,
                    analysis_only=args.analysis_only,
                )
            )
    else:
        selected_results = []
    report = evaluate_stage_one_gate(config, scenarios)
    print(json.dumps(report, indent=2))
    selected_passed = bool(selected_results) and all(
        result["gate"]["passed"] for result in selected_results
    )
    return 0 if report["passed"] or (selected_names and selected_passed) else 1
