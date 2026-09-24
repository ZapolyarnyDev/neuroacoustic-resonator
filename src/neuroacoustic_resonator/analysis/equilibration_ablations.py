from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
    EquilibrationConfigModel,
    run_controlled_equilibration_corpus,
)
from neuroacoustic_resonator.analysis.distinguishability_corpus import (
    CorpusSplitsConfig,
)

SCENARIOS = (
    "full",
    "no_warmup",
    "no_phase_damping",
    "no_metabolite_baseline",
)


class EquilibrationAblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_config: Path
    output_dir: Path
    seed_roots: tuple[int, ...] = Field(min_length=3)
    splits: CorpusSplitsConfig
    repeats: int = Field(default=2, ge=2)
    permutation_samples: int = Field(default=2000, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def validate_study(self) -> Self:
        assignments = self.splits.assignments()
        roots = [item.seed_root for item in assignments]
        if len(set(self.seed_roots)) != len(self.seed_roots):
            msg = "ablation seed roots must be unique"
            raise ValueError(msg)
        if len(set(roots)) != len(roots) or set(roots) != set(self.seed_roots):
            msg = "ablation splits must assign each seed root exactly once"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> EquilibrationAblationConfig:
        return cls.model_validate(
            yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        )


def scenario_equilibration(
    full: EquilibrationConfigModel,
    scenario: str,
) -> EquilibrationConfigModel:
    if scenario not in SCENARIOS:
        msg = f"unsupported equilibration scenario: {scenario}"
        raise ValueError(msg)
    values = full.model_dump()
    if scenario == "no_warmup":
        values["neutral_steps"] = 0
    elif scenario == "no_phase_damping":
        values["phase_damping"] = 0.0
    elif scenario == "no_metabolite_baseline":
        values["metabolite_baseline"] = None
    return EquilibrationConfigModel.model_validate(values)


def scenario_corpus_config(
    study: EquilibrationAblationConfig,
    scenario: str,
) -> ControlledEquilibrationCorpusConfig:
    base = ControlledEquilibrationCorpusConfig.from_file(study.corpus_config)
    output = study.output_dir / scenario
    values = base.model_dump(mode="python")
    values.update(
        output_dir=output,
        output_manifest=output / "manifest.json",
        output_pairs=output / "pairs.json",
        output_embeddings=output / "causal_embeddings.csv",
        output_summary=output / "summary.json",
        seed_roots=study.seed_roots,
        splits=study.splits,
        repeats=study.repeats,
        equilibration=scenario_equilibration(base.equilibration, scenario),
    )
    return ControlledEquilibrationCorpusConfig.model_validate(values)


def run_ablation_scenario(
    study: EquilibrationAblationConfig,
    scenario: str,
) -> dict[str, Any]:
    config = scenario_corpus_config(study, scenario)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return run_controlled_equilibration_corpus(
        config_path,
        permutation_samples=study.permutation_samples,
        bootstrap_samples=study.bootstrap_samples,
    )


def summarize_ablation_study(study: EquilibrationAblationConfig) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {}
    checkpoint_sets: dict[str, set[int]] = {}
    base = ControlledEquilibrationCorpusConfig.from_file(study.corpus_config)
    for name in SCENARIOS:
        config = scenario_corpus_config(study, name)
        summary = read_json_object(config.output_summary)
        regional = read_json_object(config.output_dir / "regional_causal_report.json")
        output = regional["representations"]["output"]
        evidence = read_json_object(config.output_dir / "causal_output_evidence.json")
        classification = evidence["classification"]
        expected_branches = len(study.seed_roots) * study.repeats * len(base.stimuli)
        if summary["branch_trials"] != expected_branches:
            msg = f"{name} has an incomplete branch corpus"
            raise ValueError(msg)
        if summary["causal_pairs"] != expected_branches - summary["checkpoints"]:
            msg = f"{name} has an incomplete causal pair corpus"
            raise ValueError(msg)
        if not summary["fingerprints_verified"]:
            msg = f"{name} did not verify checkpoint fingerprints"
            raise ValueError(msg)
        if evidence["design"]["seed_roots"] != sorted(study.seed_roots):
            msg = f"{name} uses unexpected seed roots"
            raise ValueError(msg)
        checkpoint_sets[name] = {
            item["field_seed"] for item in classification["by_checkpoint"]
        }
        if len(checkpoint_sets[name]) != summary["checkpoints"]:
            msg = f"{name} has inconsistent checkpoint identities"
            raise ValueError(msg)
        scenarios[name] = {
            "equilibration": scenario_equilibration(
                base.equilibration, name
            ).model_dump(),
            "branch_trials": summary["branch_trials"],
            "causal_pairs": summary["causal_pairs"],
            "checkpoints": summary["checkpoints"],
            "output_balanced_accuracy": output["balanced_accuracy"],
            "output_by_seed_root": classification["by_seed_root"],
            "output_roots_above_chance": output["roots_above_chance"],
            "output_bootstrap_ci": output["bootstrap_ci"],
            "output_permutation_p_value": output["permutation_p_value"],
            "output_stimulus_fraction": output["stimulus_fraction"],
            "output_field_seed_fraction": output["field_seed_fraction"],
            "output_stimulus_to_seed_ratio": output["stimulus_to_seed_ratio"],
            "output_separation_margin": output["separation_margin"],
            "output_cliffs_delta": output["cliffs_delta"],
            "regional_report": str(config.output_dir / "regional_causal_report.json"),
        }
    if any(seeds != checkpoint_sets["full"] for seeds in checkpoint_sets.values()):
        msg = "ablation scenarios must use the same field seeds"
        raise ValueError(msg)
    full = scenarios["full"]
    for result in scenarios.values():
        result["accuracy_difference_from_full"] = (
            result["output_balanced_accuracy"] - full["output_balanced_accuracy"]
        )
        result["margin_difference_from_full"] = (
            result["output_separation_margin"] - full["output_separation_margin"]
        )
        full_roots = {
            item["seed_root"]: item["balanced_accuracy"]
            for item in full["output_by_seed_root"]
        }
        result["accuracy_difference_by_seed_root"] = {
            str(item["seed_root"]): item["balanced_accuracy"]
            - full_roots[item["seed_root"]]
            for item in result["output_by_seed_root"]
        }
        result["eligible_for_replication"] = (
            result["output_roots_above_chance"] >= len(study.seed_roots) - 1
            and result["output_separation_margin"] > 0.0
        )
    eligible = [
        name for name in SCENARIOS if scenarios[name]["eligible_for_replication"]
    ]
    ranking = sorted(
        eligible,
        key=lambda name: (
            -scenarios[name]["output_balanced_accuracy"],
            -scenarios[name]["output_roots_above_chance"],
            -scenarios[name]["output_separation_margin"],
            SCENARIOS.index(name),
        ),
    )
    report = {
        "study_type": "exploratory_development_ablation",
        "split_labels_are_development_only": True,
        "seed_roots": list(study.seed_roots),
        "repeats": study.repeats,
        "primary_metric": "leave_one_seed_root_out_output_balanced_accuracy",
        "selection_rule": (
            "positive cross-seed margin and at least 4/5 roots above chance; "
            "then highest output accuracy, roots above chance, separation margin, "
            "and original scenario order"
        ),
        "selected_scenario": ranking[0] if ranking else None,
        "scenarios": scenarios,
        "confirmatory": False,
        "limitations": (
            "all roots were available during model development; exploratory "
            "permutation p-values and bootstrap intervals do not validate "
            "the selected scenario on unseen roots"
        ),
    }
    study.output_dir.mkdir(parents=True, exist_ok=True)
    (study.output_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run controlled equilibration ablations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_equilibration_ablations.yaml"),
    )
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    study = EquilibrationAblationConfig.from_file(args.config)
    if not args.report_only:
        for name in (args.scenario,) if args.scenario else SCENARIOS:
            run_ablation_scenario(study, name)
    if args.scenario and not args.report_only:
        return 0
    report = summarize_ablation_study(study)
    print(json.dumps(report, indent=2))
    return 0
